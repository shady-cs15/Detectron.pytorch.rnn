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


class stmm_resnet50_conv4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, F, M):
        pass


class vanilla_resnet50_conv4_red_4x_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s0_w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s0_u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s1_w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s1_u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, F, M):
        pad0 = (0, M.shape[3]%2, 0, M.shape[2]%2)
        M_red_half = self.pool(M)
        pad1 = (0, M_red_half.shape[3]%2, 0, M_red_half.shape[2]%2)
        M_red = self.pool(M_red_half)
        F_red_half = self.pool(F)
        F_red = self.pool(F_red_half)
        M_red_new = self.relu(self.w(F_red) + self.u(M_red))
        M_red_new_upsampled =  nn.functional.pad(self.unpool(M_red_new), pad1)
        M_red_half_new = self.relu(self.s0_w(F_red_half) + self.s0_u(M_red_new_upsampled))
        M_red_half_new_upsampled = nn.functional.pad(self.unpool(M_red_half_new), pad0)
        M_new = self.relu(self.s1_w(F) + self.s1_u(M_red_half_new_upsampled))
        return M_new