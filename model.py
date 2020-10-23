import torch
import torch.nn as nn
import torch.nn.functional as F

class KanjiNet(nn.Module): 
    def __init__(self, shape, num_classes):
        super(KanjiNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv2_drop = nn.Dropout2d()
        h, w = shape[:2]
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw= conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh= conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0],-1)
        # x = F.max_pool2d(x, 2)
        # print (x.size())
        x = self.fc1(x)
        x = F.softmax(self.head(x), dim=0)
        return x
