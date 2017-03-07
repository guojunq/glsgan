*## GLSGAN
Generalized Loss-Sensitive Generative Adversarial Networks (GLS-GAN)



File: glsgan.lua

Author: Guo-Jun Qi, 

guojunq@gmail.com

Date: 3/6/2017

This implements a Generalized LS-GAN (GLS-GAN). 
For details, please refer to **Appendix D** in 

**Guo-Jn Qi. Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities. arXiv:1701.06264 [[pdf](https://arxiv.org/abs/1701.06264)]**


The cost function used in this GLS-GAN implementation is a leaky rectified linear unit with a slope set in input opt. By default it is 0.2.

- **If you set slope to 0, you shall get LS-GAN;**
- **If you set slope to 1.0, you shall get WGAN.**


Please note that the GLS-GAN is proposed as our future work in the above preprint paper, but it has NOT been carefully tested yet. So please use it **at your own discretion**.

### For celebA dataset
1. Setup and download dataset 

```bash
mkdir celebA; cd celebA
```

Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".

```bash
unzip img_align_celeba.zip; cd ..
DATA_ROOT=celebA th data/crop_celebA.lua
```

2. Training the GLS-GAN

```bash
DATA_ROOT=celebA dataset=folder th gls-gan.lua
```

### For LSUN dataset

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
	
### How to display the generated images
  
To display images during training and generation, we will use the [display package](https://github.com/szym/display).

- Install it with: `luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec`
- Then start the server with: `th -ldisplay.start`
- Open this URL in your browser: [http://localhost:8000](http://localhost:8000)

### Acknowledge: 

1. parts of codes are reused from DCGAN at https://github.com/Newmu/dcgan_code


