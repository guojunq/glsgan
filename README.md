# Generalized Loss-Sensitive Generative Adversarial Networks (GLS-GAN)


File: glsgan.lua

Author: Guo-Jun Qi, 

guojunq@gmail.com


Date: 3/6/2017

This implements a Generalized LS-GAN (GLS-GAN). 
For details, please refer to **Appendix D** in 

**Guo-Jn Qi. Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities. arXiv:1701.06264 [[pdf](https://arxiv.org/abs/1701.06264)]**


The cost function used in this GLS-GAN implementation is a leaky rectified linear unit (LeakyReLU) whose slope is set in the input opt. By default it is -1.

- **If you set slope to 0, you shall get LS-GAN;**
- **If you set slope to 1.0, you shall get WGAN.**
- **If you set slope to -1.0, the cost function becomes L1 cost, i.e., C(a)=|a| and the loss function L will minimize |\Delta(real, fake)+L(real)-L(fake)|.  This is a very interesting case of GLS-GAN beyond the unknown class of GANs**
- **In the theory we showed in the preprint, slope can be set to any value in [-\infty, 1].** 

aPlease note that the GLS-GAN is proposed as our future work in the above preprint paper, but it has NOT been carefully tested yet. So please use it **at your own discretion**.

## Notes on tuning hyperparameters
The most important hyperparameter that has a direct impact on the performance is the **opt.slope** controlling the negative slope of the Leaky Linear Rectifier of the cost function. You can make a side-by-side comparison among WGAN (slope=1), LS-GAN (slope=0), and other GLS-GANs by varying opt.slope.

bWe compare the generation results by different slopes on celebA at [An incomplete map of the GAN models](http://www.cs.ucf.edu/~gqi/GANs.htm).




## For celebA dataset
1. Setup and download dataset 

```bash
mkdir celebA; cd celebA
```

Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".

```bash
unzip img_align_celeba.zip; cd ..
DATA_ROOT=celebA th data/crop_celebA.lua
```b

2. Training the GLS-GAN

GLS-GAN with C(a)=|a|
```bash
DATA_ROOT=celebA dataset=folder slope=-1 th glsgan.lua
```
GLS-GAN with C(a)=a, i.e., Wasserstein GAN
```bash
DATA_ROOT=celebA dataset=folder slope=1 th glsgan.lua
```
GLS-GAN with C(a)=(a)_+, i.e., LS-GAN
```bash
DATA_ROOT=celebA dataset=folder slope=0 th glsgan.lua
```

## For LSUN dataset

1. Please download bedroom_train_lmdb from http://lsun.cs.princeton.edu

2. Prepare the dataset following the instructions below 

  1. Install LMDB in your system: 
   	`sudo apt-get install liblmdb-dev`
	
  2. Install torch packages:
   	```
	luarocks install lmdb.torch
	luarocks install tds
	```
	
  3. Once downloading bedroom_train_lmdb, unzip the dataset and put it in a directory `lsun/train`
   
  4. Create an index file :
	Copy lsun_index_generator.lua to lsun/train, and run
	```
	cd lsun/train
	DATA_ROOT=. th lsun_index_generator.lua
	```
	Now you should have bedroom_train_lmdb_hashes_chartensor.t7 in lsun/train
	
   5. Now return to the parent direcotry of lsun, and you should be ready to run lsgan.lua:
   	```
	DATA_ROOT=lsun th glsgan.lua
	```
	
## How to display the generated images
  
To display images during training and generation, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

## Acknowledge: 

1. parts of codes are reused from DCGAN at https://github.com/Newmu/dcgan_code


