import torch
from torch import nn
import torch.nn.functional as F
from encoder import main as Encoder

USE_CUDA = True

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")


class DNNNet(nn.Module):
    def __init__(self, thought_size):
        super().__init__()

        # self.fc1 = nn.Linear(thought_size, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 2)
        self.encoder = Encoder()
        self.fc1 = nn.Linear(thought_size, 128)
        # self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        # self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)
        # self.out = nn.Sigmoid()

    def forward(self, x):

        # x = self.fc1(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2)
        #
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2)
        #
        # x = self.fc3(x)
        # x = F.sigmoid(x)
        _, x = self.encoder.encoder(x)

        x = self.fc1(x)
        # x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        # x = self.out(x)
        return x
