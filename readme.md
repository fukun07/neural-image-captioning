### Introduction
This code is used in the paper "Aligning Where to See and What to Tell: Image Captioning with Region-based Attention and Scene-specific Contexts". See our [arXiv version](http://arxiv.org/abs/1506.06272).

We provide detailed steps to reproduce the results reported in our paper. The two novel techniques are:

- scene-specific contexts: text topics of images are extraced using Latent Dirichlet Allocation (LDA). The LSTM language model is then biased by these contexts.

- region-based attention: we build our attention item-set using flexible visual regions proposed by Selective Search. The model can thus focus on more conceptually meaningful region of an image.

If you find our code or model useful in your research, please consider citing (accepted by TPAMI and now available as [pre-print](http://ieeexplore.ieee.org/document/7792748/)):

	
	@article{fu2016aligning,
  		title={Aligning where to see and what to tell: image captioning with region-based attention and scene-specific contexts},
  		author={Fu, Kun and Jin, Junqi and Cui, Runpeng and Sha, Fei and Zhang, Changshui},
  		journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on (TPAMI)},
  		year={2016}
	}

Our previous arXiv version can be found at [arXiv paper](https://arxiv.org/abs/1506.06272)



### License
This code is released under the MIT License (refer to the LICENSE file for details).


### Requirements
#### 1. software
The model is trained using Theano, a popular Python framework for fast computation of mathematical expression. To install Theano, please refer to  [Installing Theano](http://deeplearning.net/software/theano/install.html#install).

The data and trained models are stored in hdf5 format. So you also need to install [h5py](http://docs.h5py.org/en/latest/build.html) package.

The MSCOCO API is used in our code. Instead of creating a submodule, we directly place the related codes under codes/pycoco/ with some revision. We thank the MSCOCO team for their data and code. You can visit their [repository](https://github.com/tylin/coco-caption) for original code.

#### 2. hardware
Though you can run the code on CPU, we highly recommend you to equip a GPU card. A typical training process on MSCOCO takes 10h on a Nvidia K10 GPU, while it may take weeks on CPU.

The feature size for attention model is big so you need a large memory when running attention model. For the largest dataset mscoco, at least 30G memory is required (note, not GPU memory). 


### Quick start
#### 1. clone the repository
	git clone https://github.com/fukun07/neural-image-captioning.git

#### 2. download data
We provide the processed data of Flickr8K, Flickr30K and MSCOCO. You may need to wait a moment when the webpage is loading.

-  [data.zip](https://mega.nz/#!N4IEBQJC!MEmjk9QwdnjcgGgnyAd3dlJ3znkhDoPWYuXOPuGtZoQ) (0.98G) includes bounding boxes, captions, scene topics and the whole image CNN features extracted by ResNet.

After downloading the **data.zip**, you are able to run model **baseline** and **ss** (scene-specific). 

#### 3. configurate path
Please open the **config.py** script and define the path as you like:

- DATA_ROOT: where the data are placed, default to be '../data'

- SAVE_ROOT: where the trained models are saved, default to be '../saved'

#### 4.train model
Please run **train.py** under codes/ using GPU:

	THEANO_FLAGS=floatX=float32,device=gpu python train.py

or using CPU:

	THEANO_FLAGS=floatX=float32,device=cpu python train.py

Then it will start to train a model using scene-specific contexts (**ss** by default). The first compilig time may be long, you can do something else. (to me, about 15min)

In SAVE_ROOT, now you can see a new directory named **mscoco-ss-nh512-nw512-mb64-V8843**. The model is trained on MSCOCO dataset with 512 LSTM hidden size, 512 word embedding size, 64 mini-batch size and 8843 vocalbulary size.

The training will take hours to finish. If you cannot wait, you can skip to next step and use the pre-trained model provided by us. The model directory [mscoco-ss-nh512-nw512-mb64-V8843.zip](https://mega.nz/#!90wkSYwB!kIuWwplSD69vGzGDKKXiLIfhEQzrYwrqf7Kboh7X2kA) should be unzipped and then placed under SAVE_ROOT.

#### 5. generate caption
To generate captions using the trained model, please run **infer.py** under codes/ :

	THEANO_FLAGS=floatX=float32,device=gpu python infer.py

Like the training, you can choose the device as GPU or CPU. The image id and generated caption will be printed in the screen. You can visit [MSCOCO](http://mscoco.org/explore/) to see the images by searching with image id.

Bleu1-4, METEOR, ROUGE-L, CIDEr-D scores will be shown after generation process.


### Run Attention model
#### 1. download data

Attention model needs a larger feature set. For one image, ResNet is appplied to 30 regions to get a 30x2048 feature set (we call it **30res**). Since the feature files are large, we didn't include them in the **data.zip**. Instead we provide separate links to download them.

- [flickr8k-30res.zip](https://mega.nz/#!dxhgSIyR!DDGmRr-KJguHzqCg15uhAMcBLB_cVNiZXcf2WWF9btE) (1.5G) includes the 30-region ResNet features for Flickr8K. Please unzip it and place it under DATA_ROOT/flickr8k/features/.

- [flickr30k-30res.zip](https://mega.nz/#!5o5CAJwS!edH97itEWU17XIeTTZqr2EIduhRZCIvTV6ZwQZ_r6Zw) (5.8G) includes the 30-region ResNet features for Flickr30K. Please unzip it and place it under DATA_ROOT/flickr30k/features/.

- [mscoco-30res.zip](https://mega.nz/#!twRBxSIY!FQAHVFoVPVdnS-3CkVL95e4zDEl8WCQPaoUV_aYfj2A) (16.4G) includes the 30-region ResNet features for MSCOCO. Please unzip it and place it under DATA_ROOT/mscoco/features/.


We also provide an alternative source for downloading the data. Please refer to [Baidu Pan](http://pan.baidu.com/s/1c1Mtndm) in Chinese.


#### 2. or extract features by yourself

You can also extract the features using [Caffe](http://caffe.berkeleyvision.org/). We provide the original code that was used by us to extract 30res features. The general steps can be:

1. install Caffe and then set the CAFFE_ROOT in codes/config.py

2. download ResNet files from Caffe [model-zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and place the files under CAFFE_ROOT/models/ResNet/.

3. edit **cnn_feature.py** to your settings and run the code




#### 3. run the model

Now you can train and test model **ra** (region-based attention) and **rass** (region-based attention + scene-specific contexts).


### Code structure
In case you want to revise the code to run your own experiments, we describe the code design in the below.

#### 1. model/

We create a class for each of the models. Within the class, the Theano computation graph is defined and the parameters are initialized. Three members are necessary for a model class:

- self.inputs: a list of Theano tensors containing all the needed inputs.

- self.params: a list of parameters to be trained. Note that parameters not included in self.params will not be updated by optimizer.

- self.costs: a list of loss, the 0-index loss will be used as the objective function by optimizer.

These 3 members are important interfaces. All the model file are placed under codes/model/. Some common and basic layers such as mlp are defined in **layers.py**.

#### 2. reader.py

The reader is defined in **reader.py**. It reads all the needed data and manage the data feeding process. Users only need to call its method reader.iterate_batch() to get a new mini-batch of data. The reader uses multi-threading thus it can package next mini-batch when model is under training.

You can partially load a dataset by setting **head** and **tail**, the index range of images. This is very useful in debugging when you do not want the whole dataset be loaded.

#### 3. optimizer.py

The optimizers are defined in **optimizer.py**. It includes two methods: Adam and SGD. Adam got much better results on our experiments and is used by default.

The optimizer return two functions:

- train_func: compute loss and update parameters in model.params.

- valid_func: only compute loss.


#### 4. beam_search.py

Implement the beam search code. The word probability will be the ensemble reusult of a list of models. If given one model (still in python list format), it will generate sentences using single model.


#### 5. train.py

There are 4 choices for task: 

1. 'gnic': stands for Google NIC, used as base-line
2. 'ss': stands for scene-specific contexts
3. 'ra': stands for region-based attention
4. 'rass': stands for scene-specific contexts + region-based attention

and 3 choices for dataset: 'flickr8k', 'flick30k' and 'mscoco'.


#### 6. infer.py

Similar to train.py, you can choose the task(model) and dataset when inference. The variable **model_list** is a list of saved model files. If given more than one file, ensemble will be used in generation.


#### 7. scene.py

To predict LDA topic vectors using image features.


#### 8. cnn_feature.py

To extract CNN features


#### 9. tools.py

Some useful tools, such as timer and logger.


















