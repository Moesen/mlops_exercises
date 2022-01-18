import logging
import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dotenv import find_dotenv, load_dotenv
from architecture import AwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange
from dataset import CorruptMNISTDataset

# Beutification of plots
sns.set(
    rc={
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "dimgrey",
        "xtick.bottom": False,
        "xtick.color": "dimgrey",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "dimgrey",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
    },
)

@click.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("model_filename", type=click.Path())
@click.option("--lr", type=float, help="Learning Rate", default=0.003)
@click.option("--ep", type=int, help="Number of epochs", default=10)
@click.option("--dp", type=float, help="Drop percentage for dropout", default=0.5)
@click.option("--bs", type=int, help="Batchsize", default=32)
def main(train_file, model_filename, lr, ep, dp, bs):
    logging.info(f"Beginning training with {train_file} dataset.")
    logging.info(f"{lr = }, {ep = }, {dp = }")

    # Creating new model
    model = AwesomeModel.MyAwesomeModel(784, 10, [256, 128, 64], drop_p=dp)

    # Loading datasets
    train_data = CorruptMNISTDataset(train_file, train=True)
    trainloader = DataLoader(train_data, batch_size=bs, shuffle=True)

    test_data = CorruptMNISTDataset(train_file, train=False)
    testloader = DataLoader(test_data, batch_size=bs, shuffle=True)

    # NLLLoss cause we're working with probs
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For keeping track of losses
    train_losses, test_losses, test_accuracy = [], [], []
    # Live progressbar for keeping track of progress
    t = trange(ep, desc="Epoch", leave=True)

    for e in t:
        running_loss = 0
        for images, labels in trainloader:
            # Resize images as we are working with 784 tensors in a Linear network
            images = images.resize_(images.size()[0], 784)

            # Zeroing gradiant to not include results from last training
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            with torch.no_grad():
                model.eval()
                accuracy = 0
                test_loss = 0

                # Calculating accuracy on test set
                # Not optimal soluition should be validation set
                for images, labels in testloader:
                    images = images.resize_(images.size()[0], 784)
                    output = model.forward(images)
                    test_loss += criterion(output, labels).item()

                    ps = torch.exp(output)
                    equal = labels.data == ps.max(1)[1]
                    accuracy += equal.type_as(torch.FloatTensor()).mean()

            train_losses.append(float(running_loss / len(trainloader)))
            test_losses.append(float(test_loss / len(testloader)))
            test_accuracy.append(float(accuracy / len(testloader)))

            model.train()
            t.set_description(f"Acc: {accuracy/len(testloader):.2f}")

    # Adding .pth ending to filename, if not included in input
    model_filename = (
        model_filename if ".pt" in model_filename else model_filename + ".pt"
    )
    # Final path, takes into account the root of folder,
    # and creates path to trained models folder
    model_path = os.path.join(
        Path(__file__).resolve().parents[2], "models", model_filename
    )
    
    AwesomeModel.save_checkpoint(model, model_path)

    # Creating visualization of losses and stores them in reports
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Epochs: 5")
    ax1.plot(train_losses)
    ax1.plot(test_losses)
    ax1.set_title("Loss")
    ax1.legend(["Train", "Test"])

    ax2.plot(test_accuracy)
    ax2.set_title("Accuracy")
    ax2.legend(["Test"])

    filepath = os.path.join(
        project_dir, "reports", model_filename.split(".")[0] + ".png"
    )
    plt.savefig(filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()
