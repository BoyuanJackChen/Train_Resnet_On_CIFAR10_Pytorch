from functools import partial
import os
from torch.utils.data import random_split

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from main import *


def train_cifar(config, checkpoint_dir=None):
    args = FLAGS
    # ---------------------------------- main func in main.py ----------------------------------
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Parameters
    train_kwargs = {'batch_size': args.train_batch}
    test_kwargs = {'batch_size': args.test_batch}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Normalization parameters from https://github.com/kuangliu/pytorch-cifar/issues/19
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                                transform=transform_train)  # 50k
    dataset2 = datasets.CIFAR10('../data', train=False,
                                transform=transform_test)  # 10k
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = project1_model().to(device)

    # ------------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], betas=[0.9, 0.999])

    # tuning
    for epoch in range(300):
        train_loss = 0.0
        epoch_steps = 0
        correct = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        train_loss = train_loss / len(train_loader)
        accuracy = correct/len(train_loader.dataset)
        tune.report(loss=train_loss, accuracy=accuracy)
    print("Finished Training")


def tune_main(num_samples=3, max_num_epochs=4, gpus_per_trial=2):
    config = {
        "lr": tune.choice([5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    tune_main(num_samples=3, max_num_epochs=300, gpus_per_trial=0)
