import torch
import torch.nn as nn
from torch.autograd import Variable


class Key(nn.Module):
    def __init__(self):
        super(Key, self).__init__()

        # Convolutional Layers
        self.layer1a = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=(1, 12), stride=1, padding=(0, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1, 6)))
        self.layer1b = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=(1, 12), stride=1, padding=(0, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1, 6)))
        self.layer2a = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 8), stride=1, padding=(0, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1, 8)))
        self.layer2b = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1, 8), stride=1, padding=(0, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1, 8)))

        # Recurrent Layer
        self.hidden_size = 128
        self.input_size = 480
        self.i2h = nn.GRU(self.input_size, self.hidden_size, dropout=.75, num_layers=2, batch_first=True)

        # Linear Layers
        self.lin1 = nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh())
        self.lin2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh())
        self.lin3 = nn.Sequential(
            nn.Linear(32, 12),
            nn.Tanh())

    def forward(self, x_a, x_b):
        batch_size = x_a.shape[0]

        # Convolutional Layers
        x_a = x_a.view(batch_size, 1, 12, -1)
        out_a = self.layer1a(x_a)

        x_b = x_b.view(batch_size, 1, 12, -1)
        out_b = self.layer1b(x_b)

        out_a = self.layer2a(out_a)
        out_b = self.layer2b(out_b)
        out = torch.cat((out_a, out_b), 2)

        self.i2h.input_size = (out.shape[2] * out.shape[3])
        out = out.view(batch_size, -1, self.i2h.input_size)

        # Recurrent Layer
        out, self.hidden = self.i2h(out, self.hidden)
        out = out.view(out.shape[0], -1)

        # Linear Layers
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)

        return out

    def init_hidden(self, mini_batch_size):
        self.hidden = Variable(torch.zeros(2, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()


class Artist(nn.Module):
    def __init__(self):
        super(Artist, self).__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(10, affine=False),
            nn.Linear(10, 128),
            nn.Tanh(),
            nn.Dropout(0.1))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Dropout(0.1))
        self.fc3 = nn.Linear(256, 50)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class Popularity(nn.Module):
    def __init__(self):
        super(Popularity, self).__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(10, affine=False),
            torch.nn.Linear(10, 64),
            nn.Tanh(),
            nn.Dropout(0.1))
        self.fc2 = nn.Sequential(
            torch.nn.Linear(64, 128),
            nn.Tanh(),
            nn.Dropout(0.1))
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


