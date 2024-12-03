import torch
from torchvision import transforms
from Unet_model import Unet
from forward_reverse import GaussianDiffusion
import time
from PIL import Image
from define_dataset import MFDataset
from torch.utils.data import DataLoader
from img_refine_module import Restormer

class trainer(object):
    def __init__(self,
                 folder1='../test_img/multifocus_lytro', # test dataset path ("RGB")
                 folder2='../test_img/multifocus_lytro_Y', # test dataset path (“Y channel")
                 results_folder='../test_result/multifocus_lytro', # test result path
                 batch_size=1
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.results_folder = results_folder

        self.device = torch.device("cuda:0")

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = MFDataset(folder1, folder2, transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        unet = Unet().to(self.device)
        rstormer = Restormer().to(self.device)
        self.model = GaussianDiffusion(unet, rstormer).to(self.device)


    def Nomalization1(self, data):

        data = (data - data.min()) / (data.max() - data.min())
        return data

    def Nomalization2(self, data):
        return data + 0.5

    def test(self):

        dataiter = iter(self.dataloader)

        ckpt = torch.load(f'../one_try_result/ckpt_{str(100)}_.pt', map_location=self.device)
        self.model.load_state_dict(ckpt)
        print('model load weight done.', '正在反向推理生成图像，请等待...', sep='\n')

        a = time.time()  # 记录当前时间
        self.model.eval()
        with torch.no_grad():
            for i in range(1, 21):

                data1, data2, data1_y ,data2_y= next(dataiter)  # 调用各批元素(即各批批图像)
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)
                data1_y = data1_y.to(self.device)
                data2_y = data2_y.to(self.device)

                fuse_img = self.model.ddim_sample(data1=data1, data2=data2, data1_y=data1_y, data2_y=data2_y)

                for id in range(0, self.batch_size):

                    img1 = self.Nomalization2(fuse_img[id, 0:3, :, :]).mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to(
                        'cpu', torch.uint8).numpy()
                    img1 = Image.fromarray(img1)

                    if id + 1 + (i - 1) * self.batch_size < 10:
                        img1.save(f'{self.results_folder}/0{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    elif id + 1 + (i - 1) * self.batch_size < 20:
                        img1.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    elif id + 1 + (i - 1) * self.batch_size < 30:
                        img1.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    else:
                        img1.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')

                print(f'第{i}批次图像已测试完成，正在进行下一批图像测试，请等待...')
        b = time.time()
        print('测试已全部完成，耗费时间(s):', b - a)

Trainer = trainer()
Trainer.test()
