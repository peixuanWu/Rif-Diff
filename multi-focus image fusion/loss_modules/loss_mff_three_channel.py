import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.transforms.functional as TF

# from Dif-fusion
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)

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
class fusion_loss_mff_three(nn.Module):
    def __init__(self):
        super(fusion_loss_mff_three, self).__init__()
        self.L_Grad = L_Grad_three()
        self.L_Inten = L_Intensity_three()

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
class fusion_loss_mff_two(nn.Module):
    def __init__(self):
        super(fusion_loss_mff_two, self).__init__()
        self.L_Grad = L_Grad_two()
        self.L_Inten = L_Intensity_two()

    def forward(self, image_fused, image_gt):
        loss_int = 1 * self.L_Inten(image_fused, image_gt)
        loss_gradient = 1 * self.L_Grad(image_fused, image_gt)
        fusion_loss = loss_int + loss_gradient
        return fusion_loss, loss_int, loss_gradient