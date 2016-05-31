### Introduction
This code is used in the paper "Aligning Where to See and What to Tell: Image Captioning with Region-based Attention and Scene-specific Contexts". See our [arXiv version](http://arxiv.org/abs/1506.06272).

We provide detailed steps to reproduce the results reported in our paper. The two novel techniques are:

- scene-specific contexts: text topics of images are extraced using Latent Dirichlet Allocation (LDA). The LSTM language model is then biased by these contexts.

- region-based attention: we build our attention item-set using flexible visual regions proposed by Selective Search. The model can thus focus on more conceptually meaningful region of an image.

If you find our code or model useful in your research, please consider citing:

	@article{jin2015aligning,
  		title={Aligning where to see and what to tell: image caption with region-based attention and scene factorization},
  		author={Jin, Junqi and Fu, Kun and Cui, Runpeng and Sha, Fei and Zhang, Changshui},
  		journal={arXiv preprint arXiv:1506.06272},
  		year={2015}
	}



### License
This code is released under the MIT License (refer to the LICENSE file for details).


### Requirements
#### 1. software
The model is trained using Theano, a popular  Theano: A Python framework for fast computation of mathematical expression. To install Theano, please refer to  [Installing Theano](http://deeplearning.net/software/theano/install.html#install).

The data and trained models are stored in hdf5 format. So you also need to install [h5py](http://docs.h5py.org/en/latest/build.html) package.

The MSCOCO API is used in our code. Instead of creating a submodule, we directly place the related codes under codes/pycoco/ with some revision. We thank the MSCOCO team for their data and code. You can visit their [repository](https://github.com/tylin/coco-caption) for original code.

#### 2. hardware
Though you can run the code in CPU, we highly recommend you to equip a GPU card. A typical training process on MSCOCO takes 10h in a Nvidia K10 GPU, while it may take weeks in CPU.


### Run the code
#### 1. clone the repository
	git clone https://github.com/fukun07/neural-image-captioning.git

#### 2. download the data
We provide the processed data of Flickr8K, Flickr30K and MSCOCO. There are totally 4 files. You may need to wait a moment when MEGA cloud loading.

-  [data.zip 0.98G](https://mega.nz/#!N4IEBQJC!MEmjk9QwdnjcgGgnyAd3dlJ3znkhDoPWYuXOPuGtZoQ) includes bounding boxes, captions, scene topics and ResNet feature for the whole image.

- [flickr8k-30res.zip 1.47G](https://mega.nz/#!dxhgSIyR!DDGmRr-KJguHzqCg15uhAMcBLB_cVNiZXcf2WWF9btE) includes the 30-region ResNet features for Flickr8K.

- [flickr30k-30res.zip]() includes the 30-region ResNet features for Flickr30K.

- [mscoco-30res.zip]() includes the 30-region ResNet features for MSCOCO.

After downloading the **data.zip**, you are able to run model **baseline** and **ss**. So you can move on to try the code when waiting for the downloading of 30res features. (30res features are for the region-based attention models **ra** and **rass**.)

#### 3. configurate path
Please open the **config.py** script and define the path you like:

- **DATA_ROOT**: where the data are placed, default to be '../data'

- **SAVE_ROOT**: where the trained models are saved, default to be '../saved'

#### 4.train model
Please run under codes/ using GPU:

	THEANO_FLAGS=floatX=float32,device=gpu python train.py

or using CPU:

	THEANO_FLAGS=floatX=float32,device=cpu python train.py

Then it will start to train a model using scene-specific contexts (**ss**). The compilig time for the first running may be long, you can take a cup of coffer or do something else. (to me, about 15min)

In SAVE_ROOT, now you can see a new directory named **mscoco-ss-nh512-nw512-mb64-V8843**. The model is trained on MSCOCO dataset with 512 LSTM hidden size, 512 word embedding size, 64 mini-batch size and 8843 vocalbulary size.

The training will take hours to finish. If you cannot wait, you can skip to next step and use the pre-trained model provided by us. The model file [mscoco-ss-nh512-nw512-mb64-V8843.zip](https://mega.nz/#!90wkSYwB!kIuWwplSD69vGzGDKKXiLIfhEQzrYwrqf7Kboh7X2kA) should be unzipped then placed under SAVE_ROOT.

#### 5. generate caption
To generate captions using the trained model, please run under codes/ :

	THEANO_FLAGS=floatX=float32,device=gpu python train.py

Like the training, you can choose the device as GPU or CPU. The image id and generated caption will be printed in the screen. You can visit [MSCOCO](http://mscoco.org/explore/) to see the images by searching with image id.

Bleu1-4, METEOR, ROUGE-L, CIDEr-D scores will be shown after generation process.


### Revise our code
We will add description for our code design soon.













