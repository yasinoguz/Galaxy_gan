import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
#from keras.api.models import Sequential
from tensorflow.keras.models import Sequential
#from keras.src.layers import Dense,Input
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Conv2D,Conv2DTranspose,LeakyReLU,Reshape,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import  BinaryCrossentropy
from skimage.io import imread
from IPython import display
from tqdm import tqdm
import time


#Load and Prepare Dataset
img_path="./Dataset/Final Ground Truth/1.jpeg"
"""
print(os.path.exists(img_path))  # True yazmalı
read_img=tf.io.read_file(img_path)
img=tf.image.decode_jpeg(read_img,channels=3)
print(img.shape)
print(img.dtype)
print(tf.reduce_max(img))
plt.imshow(img,cmap="gray")
plt.show()
"""

BATCH_SIZE=64
INPUT_DIMS=500
NUM_IMAGES_FOR_OUTPUT_DISPLAY=9

seed_data=tf.random.normal(shape=(NUM_IMAGES_FOR_OUTPUT_DISPLAY,INPUT_DIMS))
#Preproces_Image
def preprocess_image(img_path):
    read_img=tf.io.read_file(img_path)
    img=tf.image.decode_jpeg(read_img,channels=1)
    scaled_image=(tf.cast(img,tf.float32)-127.5)/127.5
    return scaled_image

#Check pre-process image
img_sample=preprocess_image(img_path)
print(tf.reduce_min(img_sample),tf.reduce_max(img_sample))

list=tf.data.Dataset.list_files("./Dataset/Final Ground Truth/*jpeg")

#for item in list.take(5):
#    print(item.numpy())

# 1- Map preprocess function to all these file paths
dataset=list.map(preprocess_image,num_parallel_calls=tf.data.AUTOTUNE)
# 2- Shuffle the Dataset
dataset=dataset.shuffle(1000,seed=42)
# 3- Cache the Data to save time
dataset=dataset.cache()
# 4-Creates Batches of Dataset
dataset=dataset.batch(64)
# 5-Prefetch the Dataset -> Keeps the Data Readt Before the Prev Epoch is completed
dataset=dataset.prefetch(tf.data.AUTOTUNE)
"""
print(dataset)
print(dataset.take(1))
"""

batch=next(iter(dataset))

sample_image=batch[0]
print(sample_image.shape)

plt.imshow(sample_image*127.5+127.5,cmap="gray")
plt.show()


#Build the Generator Model
def build_generator_model():
    model=Sequential()
    model.add(Input(shape=500,))
    model.add(Dense(4*4*1024,use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((4,4,1024)))

    # First upsampling block using Conv2D Trasnpose
    model.add(Conv2DTranspose(512, (5, 5), padding="same", use_bias=False, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, (5, 5), padding="same", use_bias=False, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (5, 5), padding="same", use_bias=False, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), padding="same", use_bias=False, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, (5, 5), padding="same", use_bias=False, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    #Final Layer
    model.add(Conv2DTranspose(1, (5, 5), padding="same", use_bias=False, strides=(2, 2)))
    return model

#Build the Discriminator Model
def build_discriminator_model():
    model=Sequential()
    model(Input(shape=(128,128,1)))

    model.add(Conv2D(32,(5,5),padding="same",strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

     
    model.add(Conv2D(64,(5,5),padding="same",strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

     
    model.add(Conv2D(128,(5,5),padding="same",strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

     
    model.add(Conv2D(256,(5,5),padding="same",strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

     
    model.add(Conv2D(512,(5,5),padding="same",strides=(2,2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))

    return model


 
generator_model=build_generator_model()
#print(dummy_model.summary())

discriminator_model=build_discriminator_model()
#print(dummy_model2.summary())
    

"""
# Rastgele latent vektör oluştur (örneğin latent_dim=500)
latent_dim = 500
noise = tf.random.normal([1, latent_dim])  # 1 örnek üretilecek

# Generator modelinden sahte görüntü üret
generated_image = generator_model(noise, training=False)
print("Generated image shape:", generated_image.shape)

# Üretilen sahte görüntüyü Discriminator'a ver
decision = discriminator_mode(generated_image, training=False)
print("Discriminator output:", decision.numpy())

# Görsel olarak da görelim
plt.imshow(tf.squeeze(generated_image) * 127.5 + 127.5, cmap='gray')
plt.title("Generated Fake Image")
plt.axis("off")
plt.show()
"""

#Define Losses and Optimizers

cross_entropy = BinaryCrossentropy(from_logits=True)

#Discrimnator loss
def discriminator_loss(real_output,fake_output):
    d_real=cross_entropy(tf.ones_like(real_output),real_output)
    d_fake=cross_entropy(tf.zeros_like(fake_output),fake_output)
    d_loss=d_real+d_fake
    return d_loss

#Generator_Loss
def generator_loss(fake_ouput):
    g_loss=cross_entropy(tf.ones_like(fake_ouput),fake_ouput)
    return g_loss

#Define discriminators 
g_optimizer=Adam(learning_rate=1e-4)
d_optimizer=Adam(learning_rate=1e-4)

#Setup Checkpoint to Save and Restore Models
checkpoint_dir="./traininig_checkpoints"
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')

#Checkpoint
checkpoint=tf.train.Checkpoint(generator=generator_model,
                               discriminator=discriminator_model,
                               generator_optimizer=g_optimizer,
                               discriminator_optimizer=d_optimizer
                               )

def generate_and_save_images(model,epoch,test_input):
    predictions=model(test_input,training=False)

    plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(3,3,i+1)
        plt.imshow(predictions[i, :, :, :],cmap="gray")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

#Define one training step
@tf.function
def training_step(real_image_batch):
    #Define noise
    noise=tf.random.normal(shape=(BATCH_SIZE,INPUT_DIMS))

    with tf.GradientTape() as gen_tape,tf.GradientTape() as dis_tape:
        fake_image_batch=generator_model(noise,training=True)

        real_outputs=discriminator_model(real_image_batch,training=True)
        fake_outputs=discriminator_model(fake_image_batch,training=True)

        #Get the losses for gen and disc models
        gen_loss=generator_loss(fake_outputs)
        disc_loss=discriminator_loss(real_outputs,fake_outputs)

        #Calculate Gradients
        gradients_for_generator=gen_tape.gradient(gen_loss,generator_model.trainable_variables)
        gradients_for_discriminator=dis_tape.gradient(disc_loss,discriminator_model.trainable_variables)

        #Update Model Parameters
        g_optimizer.apply_gradients(zip(gradients_for_generator,generator_model.trainable_variables))
        d_optimizer.apply_gradients(zip(gradients_for_discriminator,discriminator_model.trainable_variables))

#Function to Train the GAN Model

def train(dataset,epochs):
    #Train one epoch
    for epoch in range(epochs):
        start=time.time()

        for image_batch in tqdm(dataset):
            training_step(image_batch)

        #Clear the output screen before displaying generated data   
        display.clear_output(wait=True)

        #Display the images
        generate_and_save_images(generator_model,epoch+1,seed_data)

        #Save the checkpoint after every 15 epochs
        if (epoch+1)%15==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time for {epoch+1} epoch is {time.time()-start} seconds.")


    #After the final epoch,display the outouts
    display.clear_output(wait=True)
    generate_and_save_images(generator_model,epochs,seed_data)

train(dataset,1000)    






        









