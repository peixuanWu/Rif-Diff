import os
from torch.utils import data
from PIL import Image
import cv2

class MFDataset(data.Dataset):
    def __init__(self, root_path1, transform, type="test"):
        # 该方法主要用于生成三个列表，分别装载指定文件夹下所有图片的绝对路径/相对路径

        self.transforms = transform
        self.type = type

        img_list1 = os.listdir(root_path1 + "/part1")
        img_list1.sort()

        img_list2 = os.listdir(root_path1 + "/part2")
        img_list2.sort()

        self.img_pt1_path = [root_path1 + '/part1/' + image for image in img_list1]
        self.img_pt2_path = [root_path1 + '/part2/' + image for image in img_list2]

        if self.type== "train":
            img_list3 = os.listdir(root_path1 + "/clear")
            img_list3.sort()
            self.img_pt3_path = [root_path1 + '/clear/' + image for image in img_list2]

    def normalize_to_neg_one_to_one(self, img):

        return img - 0.5

    def __getitem__(self, index):
        # 该方法主要通过index来获取初始化方法中生成的列表中的数据(即图像路径)，
        # 然后获取到指定的一对图像(源图像1、源图像2、清晰图像)，分别做预处理

        data1_1 = self.img_pt1_path[index]
        data1_2 = Image.open(data1_1)
        data1_3 = self.transforms(data1_2)
        data1_4 = self.normalize_to_neg_one_to_one(data1_3)

        data2_1 = self.img_pt2_path[index]
        data2_2 = Image.open(data2_1)
        data2_3 = self.transforms(data2_2)
        data2_4 = self.normalize_to_neg_one_to_one(data2_3)

        if self.type == "train":
            data3_1 = self.img_pt3_path[index]
            data3_2 = Image.open(data3_1)
            data3_3 = self.transforms(data3_2)
            data3_4 = self.normalize_to_neg_one_to_one(data3_3)
            return data1_4, data2_4, data3_4
        return data1_4, data2_4

    def __len__(self):
        # 该方法用于返回列表长度，即图像个数
        return len(self.img_pt1_path)