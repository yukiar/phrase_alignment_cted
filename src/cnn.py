import torch.nn as nn
import torch.nn.functional as F

# Shallow CNN
class CNN(nn.Module):
    out_channels = 1

    def __init__(self, map_embed_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, self.out_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 64 * self.out_channels, map_embed_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 64 * 64 * self.out_channels)
        x = self.fc(x)
        return x


# # Deeper CNN
# class CNN(nn.Module):
#     out_channels = 1
#
#     def __init__(self, map_embed_size):
#         super(CNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(1, self.out_channels, 3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 32 * self.out_channels, 16 * 16 * self.out_channels)
#         self.fc2 = nn.Linear(16 * 16 * self.out_channels, 8 * 8 * self.out_channels)
#         self.fc3 = nn.Linear(8 * 8 * self.out_channels, map_embed_size)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 32 * self.out_channels)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
