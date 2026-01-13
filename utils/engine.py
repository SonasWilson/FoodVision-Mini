import torch
from torch import nn, optim
# from torchvision import transforms, datasets

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # train mode
    model.train()
    # set values for train loss and accuracy
    train_loss, train_acc =0, 0

    # loop through dataloader
    for X,y in dataloader:
        # set device agnostic code
        X, y = X.to(device), y.to(device)
        # forward pass
        y_pred = model(X)

        # calculate the loss
        loss = loss_fn(y_pred, y)

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backwards
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate and accumulate loss and acc
        train_loss += loss.item()
        train_acc += (y_pred.argmax(dim=1) == y).sum().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)

    return train_loss, train_acc


def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        device: torch.device
):
    # eval model
    model.eval()

    # set test loss and acc value
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        # loop through dataloader
        for X, y in dataloader:
            # set to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            y_pred = model(X)

            # calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # calculate and accumulate accuracy
            test_acc += (y_pred.argmax(dim=1) == y).sum().item()

        # adjust to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)

    return test_loss, test_acc
