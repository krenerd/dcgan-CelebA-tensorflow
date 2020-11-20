import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import time

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
parser.add_argument("--initial_epoch", type=int,default=0)
parser.add_argument("--epoch", type=int,default=100)
parser.add_argument("--load_model", type=str2bool,default=True)
parser.add_argument("--dataset", type=str, choices=['celeba'])
parser.add_argument("--generate_image", type=str2bool,default=True)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--learning_rate_dis",type=float,default=0.000001)
parser.add_argument("--learning_rate_gen",type=float,default=0.000001)

def save_model(g,d):
    dir='./logs'
    g.save(os.path.join(dir,'generator.h5'))
    d.save(os.path.join(dir,'discriminator.h5'))
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
      
def load_celeba(batch_size):
    return tfds.load('celeb_a',data_dir='./data')['train'].batch(batch_size)

def make_folder():
    paths=['./logs','./logs/images']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
      plt.axis('off')

  plt.savefig(f'./logs/images/epoch_{epoch}.png')
  
@tf.function
def train_step(images,generator,discriminator):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    noise = tf.random.normal([args.batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(input_pipeline(images), training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
if __name__ == '__main__':
    args = parser.parse_args()
    tf.random.set_seed(42)
    #Load data
    make_folder()
    if args.dataset == 'celeba':
        print("Downloading CelebA dataset...")
        dataset=load_celeba(args.batch_size)
        print("Downloading Complete")
    
    #Build model
    print('Building model...')
    input_pipeline=model.build_input()
    
    #Load model
    if args.load_model:
        print('Loading model...')
        generator,discriminator=load_model()
    else:
        generator=model.build_generator()
        discriminator=model.build_discriminator()
        
    #Train loop
    tf.random.set_seed(42)
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    generator_optimizer = tf.keras.optimizers.Adam(args.learning_rate_gen)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.learning_rate_dis)

    for epoch in range(args.initial_epoch,args.epoch):
        start = time.time()
    
        for image_batch in progressbar.progressbar(dataset):
          train_step(image_batch['image'],generator,discriminator)

          
        # Produce images for the GIF as we go
        if args.generate_image:
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
        save_model(generator,discriminator)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    generate_and_save_images(generator,
                                 'final',
                                 seed)
    save_model(generator,discriminator)
    print("Training completed...")
    