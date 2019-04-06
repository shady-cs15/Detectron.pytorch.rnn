import torch.nn as nn
import torch

class vanilla_resnet50_conv4_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, F, M):
        M_new = self.relu(self.w(F) + self.u(M))
        return M_new


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


class stmm_resnet50_conv4_v0(nn.Module):
    # No MatchTrans, No reduced propagation
    def __init__(self):
        super().__init__()
        self.wz = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.wr = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.uz = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.ur = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, F, M):
        z_unnormalized = self.relu(self.wz(F) + self.uz(M))
        r_unnormalized = self.relu(self.wr(F) + self.ur(M))
        z_mu, z_sigma = torch.mean(z_unnormalized), torch.std(z_unnormalized)
        r_mu, r_sigma = torch.mean(r_unnormalized), torch.std(r_unnormalized)
        z_scale, r_scale = z_mu + 3*z_sigma, r_mu + 3*r_sigma
        z = z_unnormalized / z_scale
        r = r_unnormalized / r_scale
        M_cand = self.relu(self.w(F) + self.u(M*r))
        M = (1-z)*M + z*M_cand
        return M


class stmm_r50c4_rp4x_v0(nn.Module):
    # stmm with reduced propagation, no deform conv
    def __init__(self):
        super().__init__()
        self.wz = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.wr = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.uz = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.ur = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.relu = nn.ReLU()        

        self.s0_w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s0_u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s1_w = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.s1_u = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, F, M):
        M_red_half = self.pool(M)
        M_red = self.pool(M_red_half)
        F_red_half = self.pool(F)
        F_red = self.pool(F_red_half)
        pad0 = (0, M.shape[3]%2, 0, M.shape[2]%2)
        pad1 = (0, M_red_half.shape[3]%2, 0, M_red_half.shape[2]%2)

        z_unnormalized = self.relu(self.wz(F_red) + self.uz(M_red))
        r_unnormalized = self.relu(self.wr(F_red) + self.ur(M_red))
        z_mu, z_sigma = torch.mean(z_unnormalized), torch.std(z_unnormalized)
        r_mu, r_sigma = torch.mean(r_unnormalized), torch.std(r_unnormalized)
        z_scale, r_scale = z_mu + 3*z_sigma, r_mu + 3*r_sigma
        z = z_unnormalized / z_scale
        r = r_unnormalized / r_scale
        M_cand = self.relu(self.w(F_red) + self.u(M_red*r))
        M_red_new = (1-z)*M_red + z*M_cand
        
        M_red_new_upsampled =  nn.functional.pad(self.unpool(M_red_new), pad1)
        M_red_half_new = self.relu(self.s0_w(F_red_half) + self.s0_u(M_red_new_upsampled))
        M_red_half_new_upsampled = nn.functional.pad(self.unpool(M_red_half_new), pad0)
        M_new = self.relu(self.s1_w(F) + self.s1_u(M_red_half_new_upsampled))
        return M_new