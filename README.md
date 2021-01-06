# DCGAN-CelebA-tensorflow

Tensorflow implimentation of the [DCGAN](https://arxiv.org/abs/1511.06434)(Deep Convolutional Generative Adversarial Networks) model. Insipired by the official tensorflow DCGAN tutorial and the book Generative Deep Learning and its [github repository](https://github.com/davidADSP/GDL_code).

![](images/Epoch%20120.png)

Image generated at 120 epoch. 
## Paper Features
- Replace any pooling layers with strided convolutions (discriminator) and deconvolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

This implimentation of the DCGAN paper is based on the rules described above. 
But the following features varies from the original implimentation(although customizable).
- Learning rate is 10^-6 instead of 10^-4, the learning rate turned out too big and resulted mode collapse.
- Instead of the dataset proposed in the paper for facial image generation, the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) was utilized. 
## Model Training

Guide for training and inference of DCGAN-CelebA code. 

### Requirements

Install all requirements inculding Python 3.x installed.

```
!pip install -r requirements.txt
```



## Download data

Download the data using download_data.py. This execution will download the celeba dataset in ./data directory via tensorflow datasets. This step is a prerequisite for testing/training. 

```
!python download_data.py --dataset=celeba   #Dataset choice: currently only celeba is available
```


## Inference

The generate_image.py generates a matrix of (height,width) image based on the generator in ./logs/generator.h5. The created image is saved as a png image named generated_image.png.

```
!python generate_image.py --width=10	#Number of columns to generate
						  --height=10	#Number of rows to generate
```

## Training
The model can be trained customly. For training on a custom dataset, utilize model.py as a library only.The training logs, generated images and checkpoints all can be found under the ./logs folder. 

The following image describes the FID score curve while training on the cifar10 dataset. 

![](images/FID_graph.png)

```
!python train.py --epoch=100    #Epochs for training, default to 100
                    --initial_epoch=0   #initail epoch, default to 0
                    --load_model=True   #Whether to load pretrained weights in .logs/generator.h5
                    --evaluate_FID=True		#Evaluate FID after every epoch
                    --dataset=celeba    #Dataset choice: celeba or cifar10 is available
                    --generate_image=True   #Whether to generate imadges after every epoch
                    --batch_size=64     #Batch size, default to 64
                    --learning_rate_dis=0.000001    #Discriminator learning rate
                    --learning_rate_gen=0.000001    #Generator learning rate
```

## Evaluation with FID score

The model can be evaluated using the FID score and Inception Score. The FID score implimentation is based on [here](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/). Because the dataset is very large to assess the image quality on the whole dataset, we sample only a portion of the dataset. 

```
!python evaluate.py --dataset=celeba   #Dataset choice: celeba or cifar10 is available
                    --metric=fid    #Only FID and IS(Inception Score) is available
                    --samples   #Nubmer of samples to generate: default to 1000
```
DCGAN Model trained at the celeba dataset for 120 epoch with batch size=64, lr=1e-06
- FID Score: 205.86163128415788

Random Initialized Model
- FID Score: 510

DCGAN Model trained at the cifar10 dataset for 100 epoch with batch size=16, lr=1e-04
- FID Score: 98.4815200644252

Random Initialized Model
- FID Score: 451.4736

