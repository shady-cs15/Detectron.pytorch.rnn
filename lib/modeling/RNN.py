import torch.nn as nn

class vanilla_resnet50_conv4_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, F, M):
        M_new = self.relu(self.w(F) + self.u(M))
        return M_new


'''
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

class resnet_conv5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(4096, 2048, 3, 1, 1)
        self.conv12 = nn.Conv2d(2048, 2048, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv21 = nn.Conv2d(2048, 2048, 3, 1, 1)
        self.conv22 = nn.Conv2d(2048, 2048, 3, 1, 1)
        
    def forward(self, X):
        X = self.conv11(X)
        X = self.conv12(X)
        X = self.relu(X)
        X = self.conv21(X)
        X = self.conv22(X)
        X = self.relu(X)
        return X 
'''