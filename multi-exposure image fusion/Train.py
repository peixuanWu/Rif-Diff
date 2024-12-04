import torch
from torchvision import transforms
from define_dataset import MFDataset
from torch.utils.data import DataLoader

from Unet_model import Unet

from img_refine_module import Restormer

from forward_reverse import GaussianDiffusion
from tqdm.auto import tqdm
import time

import torchvision


loss_iters = []
loss_items_eps = []

class trainer(object):
    def __init__(self,
                 folder='../train_dataset/tif_Swin_Y',
                 results_folder='../one_try_result',
                 batch_size=10,
                 train_lr=1e-4,
                 adam_betas=(0.9, 0.99),
                 train_steps=400,
                 sample_every=20,
                 ):
        super().__init__()

        self.train_steps = train_steps
        self.save_and_sample_every = sample_every
        self.batch_size = batch_size
        self.results_folder = results_folder

        self.device = torch.device("cuda:0")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = MFDataset(folder, transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True,num_workers=0)

        unet = Unet().to(self.device)
        rstormer = Restormer().to(self.device)
        self.model = GaussianDiffusion(unet, rstormer).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        self.cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.train_steps,
                                                               eta_min=1e-6,
                                                               last_epoch=-1)
    def Nomalization(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def train(self):

        disabled = True

        a = time.time()

        self.model.train()
        for i in range(1, self.train_steps + 1):

            running_loss = 0
            running_loss_eps = 0
            running_loss_int = 0

            j=0
            with tqdm(self.dataloader, dynamic_ncols=True, disable=disabled) as tqdmDataLoader:

                for data1, data2, datar in tqdmDataLoader:

                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    datar = datar.to(self.device)

                    loss, loss_eps, loss_int = self.model.forward(data1, data2, datar)
                    running_loss += loss
                    running_loss_eps += loss_eps
                    running_loss_int += loss_int

                    self.optimizer.zero_grad()  # 清空过往梯度

                    loss.backward()  # 反向计算梯度

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 裁剪梯度，防止梯度爆炸

                    self.optimizer.step()  # 更新参数

                    tqdmDataLoader.set_postfix(ordered_dict={  # 设置进度条显示内容
                        "LR": self.optimizer.state_dict()['param_groups'][0]["lr"]
                    })

                    j = j + 1

            print(f'第{i}次训练完成，loss: {running_loss:.4f}，loss_eps: {running_loss_eps:.4f}，loss_int: {running_loss_int:.4f}')

            # 指定训练次数后进行验证
            if i % self.save_and_sample_every == 0:
                torch.save(self.model.state_dict(), f'{self.results_folder}/ckpt_{str(i)}_.pt')

            self.cosineScheduler.step()  # 学习率更新

        b = time.time()  # 记录训练完之后的时间
        print('当前训练已耗费时间(s):', b - a)  # 输出训练耗费的时间，单位s

Trainer = trainer()  # 训练函数实例化
Trainer.train()
