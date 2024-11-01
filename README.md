# *****
This is official Pytorch implementation of "********"

## Framework
The framework of the proposed ****** for improving image fusion based on diffusion model.

## update
#### 1.2024.10.14: 
We have released some code, and all the code will be released after the paper is received. Coming soon.

## Environment
 - [x] python 3.8.10
 - [x] torch 2.0.0
 - [x] torchvision 0.15.1
 - [x] numpy 1.24.2
 - Recommend：The requirements for the environment in PyTorch are not strict; you can freely configure it according to your needs and circumstances.

## Visual Comparison

#### 1. for MFF task

#### 2. for MEF task

#### 3. for IVF task

## To Test
### 1. Pretrain Weights
We provide the pretrain weights for multi-focus image fusion、multi-exposure image fusion、infrared and visible image fusion. Download the weight and put it into the corresponding folder.

The pretrain weight for multi-focus image fusion is at [Baidu Drive](https://pan.baidu.com/s/14C7S3gImgB8BCecZxyb4jQ?pwd=jf16) (code: jf16).

The pretrain weight for multi-exposure image fusion is at [Baidu Drive](https://pan.baidu.com/s/1_g0EnQwq6QP-8BVCA1anQA?pwd=7aq7) (code: 7aq7).

The pretrain weight for infrared and visible image fusion is at  [Baidu Drive](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) (code: ep8v).

### 2. test dataset
Download the test dataset from [**Lytro dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for MFF task, and put it in **./test_img/lytro/**. 

Download the test dataset from [**MEFB dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for MEF task, and put it in **./test_img/MEFB/**.

Download the test dataset from [**TNO dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for IVF task, and put it in **./test_img/TNO/**.

Download the test dataset from [**MSRS dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for IVF task, and put it in **./test_img/MSRS/**.

- Recommend：You may employ the data dataset offered by us or utilize your own data dataset.

### 3. Prepare Your Dataset
    test_img/
           part1/
           part2/
For multi-focus image fusion, part1：near focus image，part2：far focus image.

For multi-exposure image fusion, part1：low exposure image，part2：high exposure image.

For infrared and visible image fusion, part1：ir image，part2：vis image.



### 4. model parameters
All the parameters involved in the test are set up and you can use them directly

## To Train 
### 1.train dataset
Download the train dataset from [**WHU-MFI dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for MFF task, and put it in **./train_dataset/l/WHU-MFI**. 

Download the train dataset from [**MEFB dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for MEF task, and put it in **./train_dataset/MEFB/**.

Download the train dataset from [**MSRS dataset**](https://pan.baidu.com/s/1XyRdu1ZXBvvKhmROjzmYdg?pwd=ep8v) for IVF task, and put it in **./train_dataset/MSRS/**.

- Recommend：For multi-exposure image fusion, infrared and visible image fusion, we recommend that you use prior image fusion to make your own training dataset.

### 2. Prepare Your Dataset
    train_dataset/
           part1/
           part2/
For multi-focus image fusion, part1：near focus image，part2：far focus image.

For multi-exposure image fusion, part1：low exposure image，part2：high exposure image.

For infrared and visible image fusion, part1：ir image，part2：vis image.

## Citation


## Acknowledgement
