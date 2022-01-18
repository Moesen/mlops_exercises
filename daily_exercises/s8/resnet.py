import torch
import torchvision.models as models

if __name__ == "__main__":
    resnet = models.resnet18(pretrained=True)
    script_model = torch.jit.script(resnet)
    script_model.save("deployable_model.pt")
