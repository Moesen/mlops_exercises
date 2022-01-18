import logging
import os
import sys
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv

project_dir = Path(__file__).resolve().parents[2]
sys.path.append(os.path.join(project_dir, "src/models"))
print(sys.path)

from ModelArchitecture import AwesomeModel


def get_feature(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


@click.command()
@click.argument("model_filename", type=click.Path(exists=True))
@click.argument("data_filename", type=click.Path(exists=True))
def main(model_filename, data_filename):
    logging.info("Producing visualization of network")
    model = AwesomeModel.load_checkpoint(model_filename)
    data = torch.load(data_filename)
    model.global_pool.register_forward_hook(get_features("feats"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
