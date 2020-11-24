import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import numpy as np

import model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Define arguments 
parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--metric", type=str,default='fid')
parser.add_argument("--dataset", type=str, choices=['celeba'])

def load_model():
    dir='./logs'
    try:
        g=tf.keras.models.load_model(os.path.join(dir,'generator.h5'))
        d=tf.keras.models.load_model(os.path.join(dir,'discriminator.h5'))
        return g,d
    except:
        print('No appropriate weight file...')
        g=model.build_generator()
        d=model.build_discriminator()
        return g,d
      
def load_celeba():
    return tfds.load('celeb_a',data_dir='./data')['train']


def generate_and_save_images(model, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
      plt.axis('off')

  plt.savefig(f'./final_image.png')

def calculate_fid_score(gen_image,true_images):
  inception_model=tf.keras.applications.InceptionV3()
  preprocess_image=tf.keras.applications.inception_v3.preprocess_input
  num_images=image.shape[0]
  scores=[]
  n_part=num_images//n_split
  #Split the process to n_split mini batches
  preprocessed_gen=preprocess_image(gen_image)
  preprocessed_real=preprocess_image(true_images)
  act1=inception_model.predict(preprocessed_gen)
  act2=inception_model.predict(preprocessed_real)

  mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
            # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
            # calculate sqrt of product between cov
  covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
            # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
    covmean = covmean.real
            # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

  return fid
    
if __name__ == '__main__':
    args = parser.parse_args()
    tf.random.set_seed(42)
    #Load data
    if args.dataset == 'celeba':
        print("Downloading CelebA dataset...")
        dataset=load_celeba(args.batch_size)
        print("Downloading Complete")
    
    #Build model
    input_pipeline=model.build_input()
    
    print('Loading model...')
    generator=load_model()
        
    #Evaluate with FID
    num_examples_to_generate=1000
    noise_dim=100
    
    noise=tf.random.normal([num_examples_to_generate, noise_dim])
    gen_image=generator(noise,training=False)
    true_image=tfds.as_numpy(dataset)
