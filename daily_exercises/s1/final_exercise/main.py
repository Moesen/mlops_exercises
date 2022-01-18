import argparse
import json
import sys
from os import path

import data
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument(
            "--lr", default=0.1, type=float, help="Learning Rate for optimizer"
        )
        parser.add_argument("--ep", default=10, type=int, help="Number of epochs")
        parser.add_argument("--fn", default="temp", type=str, help="Filename")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel(784, 10, [256, 128, 64], drop_p=0.2)
        traindataset = data.CorruptedMNISTDataset(train=True)
        trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)

        testdataset = data.CorruptedMNISTDataset(train=False)
        testloader = DataLoader(testdataset, batch_size=32, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        epochs = args.ep
        train_losses, test_losses, test_accuracy = [], [], []

        t = trange(epochs, desc="Num of epochs", leave=True)
        for e in t:
            running_loss = 0
            for images, labels in trainloader:
                optimizer.zero_grad()
                images = images.resize_(images.size()[0], 784)
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

        checkpoint = {
            "input_size": 784,
            "output_size": 10,
            "hidden_layers": [each.out_features for each in model.hidden_layers],
            "state_dict": model.state_dict(),
        }

        filename = args.fn
        torch.save(checkpoint, path.join("model", filename + ".pth"))
        with open("logs/temp.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "train_losses": train_losses,
                        "test_loss": test_losses,
                        "test_accuracy": test_accuracy,
                    },
                    indent=2,
                )
            )

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="model/temp.pth", type=str)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = self.load_checkpoint(args.load_model_from)
        criterion = nn.NLLLoss()

        testdataset = data.CorruptedMNISTDataset(train=False)
        testloader = DataLoader(testdataset, batch_size=32, shuffle=True)

        accuracy = 0
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images = images.resize_(images.size()[0], 784)
                output = model.forward(images)
                test_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                equal = labels.data == ps.max(1)[1]
                accuracy += equal.type_as(torch.FloatTensor()).mean()
        print(
            f"Test Accuracy: {accuracy/len(testloader)}, TestLoss: {test_loss/len(testloader)}"
        )

    @staticmethod
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = MyAwesomeModel(
            checkpoint["input_size"],
            checkpoint["output_size"],
            checkpoint["hidden_layers"],
        )
        model.load_state_dict(checkpoint["state_dict"])

        return model


if __name__ == "__main__":
    TrainOREvaluate()
