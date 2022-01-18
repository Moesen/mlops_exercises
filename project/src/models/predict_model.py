import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import find_dotenv, load_dotenv
from architecture import AwesomeModel


@click.command()
@click.argument("model_filename", type=click.Path(exists=True))
@click.argument("data_filename", type=click.Path(exists=True))
@click.option("--num", type=int, default=10, help="Number of images to predict")
def main(model_filename, data_filename, num):
    logging.info(
        f"Predicting {num} random images from {data_filename} using {model_filename}"
    )
    model = AwesomeModel.load_checkpoint(model_filename)
    data = torch.load(data_filename)

    imgs = data["test_x"]
    labels = data["test_Y"]

    indices = torch.randperm(len(imgs))[:num]
    x = imgs[indices]
    Y = labels[indices]

    model.eval()
    output = model.forward(x.resize_(num, 784))
    ps = torch.exp(output)
    equal = Y.data == ps.max(1)[1]
    predictions = ps.max(1)[1].tolist()
    accuracy = equal.type_as(torch.FloatTensor()).mean()

    print(predictions)

    x.resize_(num, 28, 28)
    fig = plt.figure(figsize=(10, 10))
    prev = None
    for i, (img, label, pred) in enumerate(zip(x, Y, predictions)):
        ax = fig.add_subplot(1, num + 1, i + 1, sharey=prev, sharex=prev)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{pred=}")
        plt.subplots_adjust(wspace=0.1)
        prev = ax
    plt.add_subplot

    plt.show()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
