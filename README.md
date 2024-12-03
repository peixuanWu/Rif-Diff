This is official Pytorch implementation of "Rif-Diff: Improving image fusion based on diffusion model via residual prediction"

## Update
#### 1. 2024.12.1: 
We are submitting this paper and have released some code. All the code will be released after the paper is received. Good luck to me!

## Environment
 - [x] python 3.8.10
 - [x] torch 2.0.0
 - [x] torchvision 0.15.1
 - [x] numpy 1.24.2
 
   ......
 - Recommend：The requirements for the environment in PyTorch are not strict and you can freely configure it according to your needs and circumstances.

## To test
### 1. Pretrain Weights
We provide the pretrain weights for multi-focus image fusion, multi-exposure image fusion, and infrared and visible image fusion. Download the weights and put them into the corresponding folder.

The pretrain weights are at [Baidu Drive](https://pan.baidu.com/s/1my79LIUVnW2uwN8iAIU1PQ?pwd=zt4t) (code:zt4t).

### 2. Test dataset
We provide the test datasets for multi-focus image fusion, multi-exposure image fusion, and infrared and visible image fusion. Download and put them into the corresponding folders.

The test datasets are at [Baidu Drive](https://pan.baidu.com/s/1hSLfkC5YurIQvi8yvqxX-w?pwd=hck4) (code:hck4). 

- Recommend：You may utilize your own test dataset to test our model and perform comparison experiments.
- Note：Our datasets are derived from the widely used public datasets, which has been cited in our paper.

### 3. Prepare your dataset
    test_img/
           part1/
           part2/
For multi-focus image fusion, part1：near-focus image，part2：far-focus image.

For multi-exposure image fusion, part1：low exposure image，part2：over exposure image.

For infrared and visible image fusion, part1：ir image，part2：vis image.

### 4. Model parameters
All the parameters involved in the test are set up and you can use them directly.

### 5. Y channel
For multi-exposure image fusion and infrared and visible image fusion, our model uses the Y channel of the YCbCr color space for fusion, while the Cb and Cr channels are fused in the traditional method.

For multi-focus image fusion, we directly fuse RGB images, but the Y channel image is needed to construct the dual-step decision module.
## To train 
### 1.train dataset
We provide the train datasets for multi-focus image fusion, multi-exposure image fusion, and infrared and visible image fusion. Download and put them into the corresponding folders.

The train datasets are at [Baidu Drive](https://pan.baidu.com/s/1-xRZTi6x142EZOojRumawA?pwd=c4mg)(code:c4mg). 

- Recommend：For fusion tasks lacking the ground truth, you may use the folder "image_fusion_prior" to make your own training dataset.
- Note：Our datasets are derived from the widely used public datasets, which has been cited in our paper.

### 2. Model parameters
Most of the parameters have already been pre-configured. The hyper-parameters involved in the training process can be set according to those specified in the paper, or adjusted freely based on your own needs.


## Citation
If you find our work useful for your research, please cite our paper.

