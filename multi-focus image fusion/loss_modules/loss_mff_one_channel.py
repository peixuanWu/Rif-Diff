import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF
from loss_modules.loss_ssim import ssim

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

# three input
class L_Grad_three(nn.Module):
    def __init__(self):
        super(L_Grad_three, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

# three input
class L_Intensity_three(nn.Module):
    def __init__(self):
        super(L_Intensity_three, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        image_A = image_A.unsqueeze(0)
        image_B = image_B.unsqueeze(0)
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

# three input
class L_SSIM_three(nn.Module):
    def __init__(self):
        super(L_SSIM_three, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM

# three input
class fusion_loss_mff_three(nn.Module):
    def __init__(self):
        super(fusion_loss_mff_three, self).__init__()
        self.L_Grad = L_Grad_three()
        self.L_Inten = L_Intensity_three()
        self.L_SSIM = L_SSIM_three()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 2 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 2 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 1 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

# two input
class L_Grad_two(nn.Module):
    def __init__(self):
        super(L_Grad_two, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_fused, image_gt):
        gradient_fused = self.sobelconv(image_fused)
        gradient_gt = self.sobelconv(image_gt)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_gt)
        return Loss_gradient

# two input
class L_Intensity_two(nn.Module):
    def __init__(self):
        super(L_Intensity_two, self).__init__()

    def forward(self, image_fused, image_gt):
        Loss_intensity = F.l1_loss(image_fused, image_gt)
        return Loss_intensity

# two input
class L_SSIM_two(nn.Module):
    def __init__(self):
        super(L_SSIM_two, self).__init__()

    def forward(self, image_fused, image_gt):
        Loss_SSIM = ssim(image_fused, image_gt)
        return Loss_SSIM

# two input
class fusion_loss_mff_two(nn.Module):
    def __init__(self):
        super(fusion_loss_mff_two, self).__init__()
        self.L_Grad = L_Grad_two()
        self.L_Inten = L_Intensity_two()
        self.L_SSIM = L_SSIM_two()

    def forward(self, image_fused, image_gt):
        loss_int = 2 * self.L_Inten(image_fused, image_gt)
        loss_gradient = 2 * self.L_Grad(image_fused, image_gt)
        loss_SSIM = 1 * (1 - self.L_SSIM(image_fused, image_gt))
        fusion_loss = loss_int + loss_gradient + loss_SSIM
        return fusion_loss, loss_int, loss_gradient, loss_SSIM