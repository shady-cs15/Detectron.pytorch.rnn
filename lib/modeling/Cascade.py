import torch.nn as nn

def fpn_conv4():
    return fpn_convX()
    

class fpn_convX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 1)
        
    def forward(self, X):
        X = self.conv11(X)
        X = self.conv12(X)
        X = self.relu(X)
        X = self.conv21(X)
        X = self.conv22(X)
        return X 