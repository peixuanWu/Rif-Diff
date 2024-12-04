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

                 folder1='../test_img/multiexposure_MEFB_Y',
                 results_folder='../test_result/multiexposure_MEFB_Y',

                 batch_size=1  # 每批图像个数，确保图像总数能将其除尽

                 ):
        super().__init__()

        self.batch_size = batch_size
        self.results_folder = results_folder

        # 创建训练对象
        self.device = torch.device("cuda:0")
        ############### 数据集相关操作
        # 定义预处理操作
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # 实例化数据集函数(获取数据集)、加载数据集
        dataset = MFDataset(folder1,  transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        ############### 模型相关操作
        # 实例化
        unet = Unet().to(self.device)
        rstormer = Restormer().to(self.device)
        self.model = GaussianDiffusion(unet, rstormer).to(self.device)


    def Nomalization1(self, data):

        data = (data - data.min()) / (data.max() - data.min())
        return data

    def Nomalization2(self, data):
        return data + 0.5

    def test(self):

        # 通过生成迭代器，获取一批测试图像````````````
        dataiter = iter(self.dataloader)

        # 将训练好的参数加载到Unet模型
        ckpt = torch.load(f'../one_try_result/ckpt_{str(400)}_.pt', map_location=self.device)
        self.model.load_state_dict(ckpt)
        print('model load weight done.', '正在反向推理生成图像，请等待...', sep='\n')

        a = time.time()  # 记录当前时间
        self.model.eval()
        with torch.no_grad():
            for i in range(1, 26):

                data1, data2 = next(dataiter)  # 调用各批元素(即各批批图像)
                data1 = data1.to(self.device)  # 注意，数据的取值范围为[-1,1]，数据类型为张量型
                data2 = data2.to(self.device)

                res_img, fused_img, refine_img = self.model.ddim_sample(data1=data1, data2=data2)

                for id in range(0, self.batch_size):

                    img1 = self.Nomalization1(res_img[id, 0, :, :]).mul(255).add_(0.5).clamp_(0, 255).to(
                        'cpu', torch.uint8).numpy()
                    img1 = Image.fromarray(img1)  # 将数组转为图像格式

                    img2 = self.Nomalization2(fused_img[id, 0, :, :]).mul(255).add_(0.5).clamp_(0, 255).to(
                        'cpu', torch.uint8).numpy()
                    img2 = Image.fromarray(img2)  # 将数组转为图像格式

                    img3 = self.Nomalization2(refine_img[id, 0, :, :]).mul(255).add_(0.5).clamp_(0, 255).to(
                        'cpu', torch.uint8).numpy()
                    img3 = Image.fromarray(img3)  # 将数组转为图像格式

                    # 选择命名方式，确保生成的图像的顺序跟测试的图像顺序一致,至多39幅图像
                    if id + 1 + (i - 1) * self.batch_size < 10:
                        img1.save(f'{self.results_folder}/res-0{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img2.save(f'{self.results_folder}/initial-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img3.save(f'{self.results_folder}/0{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    elif id + 1 + (i - 1) * self.batch_size < 20:
                        img1.save(f'{self.results_folder}/res-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img2.save(f'{self.results_folder}/initial-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img3.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    elif id + 1 + (i - 1) * self.batch_size < 30:
                        img1.save(f'{self.results_folder}/res-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img2.save(f'{self.results_folder}/initial--{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img3.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                    else:
                        img1.save(f'{self.results_folder}/res-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img2.save(f'{self.results_folder}/initial-{str(id + 1 + (i - 1) * self.batch_size)}.jpg')
                        img3.save(f'{self.results_folder}/{str(id + 1 + (i - 1) * self.batch_size)}.jpg')

                print(f'第{i}批次图像已测试完成，正在进行下一批图像测试，请等待...')
        b = time.time()
        print('测试已全部完成，耗费时间(s):', b - a)

Trainer = trainer()  # 训练函数实例化
Trainer.test()
