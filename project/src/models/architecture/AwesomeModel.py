import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def save_checkpoint(model: MyAwesomeModel, savepath: str):
    """Saves a model

    Args:
        model (MyAwesomeModel): The trained model
        savepath (str): The path where it should be stored
    """
    checkpoint = {
        "input_size": 784,
        "output_size": 10,
        "hidden_layers": [each.out_features for each in model.hidden_layers],
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, savepath)


def load_checkpoint(modelpath: str) -> MyAwesomeModel:
    """Loads a pretrained model

    Args:
        modelpath (str): path to the model

    Returns:
        MyAwesomeModel: Returns the trained model
    """
    checkpoint = torch.load(modelpath)
    model = MyAwesomeModel(
        checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_layers"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
