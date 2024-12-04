import torch
from torchvision import transforms
from define_dataset import MFDataset
from torch.utils.data import DataLoader

from Unet_model import Unet

from img_refine_module import Restormer_fn

from forward_reverse import GaussianDiffusion
from tqdm.auto import tqdm
import time

import torchvision


class trainer(object):
    def __init__(self,
                 folder='../train_dataset/MSRS-version2',  # 训练集路径
                 results_folder='../one_try_result',  # 存放训练结果的路径
                 batch_size=10,
                 train_lr=1e-4,
                 adam_betas=(0.9, 0.99),  #
                 train_steps=200,
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
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

        unet = Unet().to(self.device)
        rstormer_fn = Restormer_fn().to(self.device)
        self.model = GaussianDiffusion(unet, rstormer_fn).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        self.cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.train_steps,
                                                               eta_min=1e-6,
                                                               last_epoch=-1)
    def Nomalization(self, data):
        data = (data - data.min()) / (data.max() - data.min())
        return data

    def train(self):

        disabled = False

        a = time.time()

        self.model.train()
        for i in range(1, self.train_steps + 1):

            running_loss = 0
            running_loss_eps = 0

            with tqdm(self.dataloader, dynamic_ncols=True, disable=disabled) as tqdmDataLoader:

                for data1, data2, datar in tqdmDataLoader:

                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    datar = datar.to(self.device)

                    loss, loss_eps = self.model.forward(data1, data2, datar)
                    running_loss += loss
                    running_loss_eps += loss_eps
                    self.optimizer.zero_grad()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 裁剪梯度，防止梯度爆炸

                    self.optimizer.step()

                    tqdmDataLoader.set_postfix(ordered_dict={  # 设置进度条显示内容
                        "LR": self.optimizer.state_dict()['param_groups'][0]["lr"]
                    })

            print(f'第{i}次训练完成，loss: {running_loss:.4f}，loss_eps: {running_loss_eps:.4f}')


            if i % self.save_and_sample_every == 0:
                torch.save(self.model.state_dict(), f'{self.results_folder}/ckpt_{str(i)}_.pt')
            self.cosineScheduler.step()

        b = time.time()
        print('当前训练已耗费时间(s):', b - a)

Trainer = trainer()
Trainer.train()
