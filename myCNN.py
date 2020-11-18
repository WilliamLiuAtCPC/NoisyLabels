import torch.nn as nn

class CNN_NET(nn.Module):
    def __init__(self):
        super().__init__()
        '''输入为3*32*32，尺寸减半是因为池化层'''
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 输出为32*32*64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # 输出为16*16*128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 196, 3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 196, 3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU()
        )  # 输出为8*8*196
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = nn.Softmax(x)
        return x