import torch.nn as nn
from torch.autograd import Function


class DigitsNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.n_embedding = 32 * 5 * 5

        self.fc = nn.Sequential(
            nn.Linear(self.n_embedding, 32),
            nn.ReLU(),
            nn.Linear(32, args.n_classes),
        )

    def forward(self, xs):
        hs = self.encoder(xs)
        hs = hs.view(hs.size(0), -1)

        logits = self.fc(hs)
        return hs, logits


class DigitsEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.embed_size = 32 * 5 * 5

    def forward(self, xs):
        hs = self.encoder(xs)
        return hs
