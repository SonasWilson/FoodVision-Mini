import os

import torch
from torch import classes

from utils.data_setup import create_dataloader
from utils.data_download import download_data
from utils.engine import train_step, test_step
from demo.foodvision_mini.model import create_model, get_transform

def main():
    # setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    download_data()

    # setup dirs
    train_dir = "data/train"
    test_dir = "data/test"

    train_dataloader, test_dataloader, class_names = create_dataloader(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=get_transform(),
        batch_size=32
    )

    model = create_model(num_classes=len(class_names)).to(device)

    # define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)

    # epochs
    epochs = 5

    # track accuracy
    best_test_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch+1}/{epochs}: | "
              f"Train acc: {train_acc:.3f} | "
              f"Train loss: {train_loss:.3f} | "
              f"Test acc: {test_acc:.3f} | ",
              f"Test loss: {test_loss:.3f}")

        # save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "models/best_effnetb2_feature_extractor.pth")

            print(f"New best model saved with test acc: {best_test_acc:.3f}")

    print(f"Best accuracy achieved: {best_test_acc:.3f}")

if __name__ == "__main__":
    main()


