import os
from torch.utils import data
from PIL import Image
import cv2

class MFDataset(data.Dataset):
    def __init__(self, root_path1, root_path2, transform, type="test"):
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
        else:
            img_list1_y = os.listdir(root_path2 + "/part1")
            img_list1_y.sort()

            img_list2_y = os.listdir(root_path2 + "/part2")
            img_list2_y.sort()

            self.img_pt1_path_y = [root_path2 + '/part1/' + image for image in img_list1_y]
            self.img_pt2_path_y = [root_path2 + '/part2/' + image for image in img_list2_y]

    def normalize_to_neg_one_to_one(self, img):

        return img - 0.5

    def rgb2y(self, img):
        img_r = img[:, :, 0]
        img_g = img[:, :, 0]
        img_b = img[:, :, 0]
        img_y = 0.257 * img_r + 0.564 * img_g + 0.098 * img_b + 16
        return img_y

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
        else:

            # 方式一
            # opencv读入的矩阵是BGR，转为RGB
            # img_cv2(BGR) --> img_rgb(RGB)

            # data1_y_1 = cv2.imread(self.img_pt1_path[index])
            # data1_y_2 = cv2.cvtColor(data1_y_1, cv2.COLOR_BGR2RGB)
            # data1_y_3 = self.rgb2y(data1_y_2)/255
            # data1_y_4 = self.transforms(data1_y_3)
            # data1_y = self.normalize_to_neg_one_to_one(data1_y_4)
            #
            # data2_y_1 = cv2.imread(self.img_pt2_path[index])
            # data2_y_2 = cv2.cvtColor(data2_y_1, cv2.COLOR_BGR2RGB)
            # data2_y_3 = self.rgb2y(data2_y_2)/255
            # data2_y_4 = self.transforms(data2_y_3)
            # data2_y = self.normalize_to_neg_one_to_one(data2_y_4)

            # 方式二：
            data1_1_y = self.img_pt1_path_y[index]
            data1_2_y = Image.open(data1_1_y)
            data1_3_y = self.transforms(data1_2_y)
            data1_y = self.normalize_to_neg_one_to_one(data1_3_y)

            data2_1_y = self.img_pt2_path_y[index]
            data2_2_y = Image.open(data2_1_y)
            data2_3_y = self.transforms(data2_2_y)
            data2_y = self.normalize_to_neg_one_to_one(data2_3_y)

        return data1_4, data2_4, data1_y ,data2_y

    def __len__(self):
        return len(self.img_pt1_path)