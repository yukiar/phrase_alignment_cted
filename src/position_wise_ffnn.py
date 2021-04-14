import torch.nn as nn


# Position-wise FFNN
class Position_wise_FFNN(nn.Module):

    def __init__(self, embed_size):
        super(Position_wise_FFNN, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.fc2(self.activ(self.fc1(x)))
        return x
