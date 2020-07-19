import tensorflow as tf
import logging
import time
from losses import disc_loss, gen_en_loss
from misc import get_fixed_random, generate_images
import random
#from Model1 import Model1

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from tensorflow.keras import layers
from collections import defaultdict
import time
from keras.layers import Input
from IPython import display
from keras.models import Model
def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled
def layer_names(model):
    layer_to_compute = []
    for layer in model.layers:
        if layer.name is not 'flatten' and 'input':
            layer_to_compute.append(layer.name)
    return layer_to_compute
def compose_layer_model(model,layer_to_compute):
    
    layer_outputs = [model.get_layer(layer_name).output for layer_name in layer_to_compute]
    layer_model = Model(input=model.input,
                    output=layer_outputs)
    return layer_model
def set_cov_dict(model):
    out_dict = defaultdict(list)
    for layer in model.layers:
        if layer.name is not 'flatten' and 'input':
            for neuron_idx in range(layer.output_shape[-1]):
                out_dict[(layer.name,neuron_idx)] = False
    return out_dict

def update_coverage(input,layer_model,threshold,layer_name,out_dict):
    activations = layer_model.predict(np.expand_dims(input,axis=0))
    name=0
    for layer_output in activations:
        scaled = scale(layer_output)
        for neuron_idx in range(layer_output.shape[-1]):
            if np.mean(scaled[..., neuron_idx ]) is not None:
                if np.mean(scaled[..., neuron_idx ]) > threshold and not out_dict[(layer_name[name],neuron_idx)]:
                    out_dict[(layer_name[name],neuron_idx)] = True
        name = name + 1
def neuron_cov(out_dict):

    covered_neurons = len([v for v in out_dict.values() if v])
    total_neurons = len(out_dict)

    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)
def cifar10vgg(input_tensor=None, train=False):
    num_classes = 10
    weight_decay = 0.0005
    x = Conv2D(64, (3, 3), padding='same',name='conv_1',
                  input_shape=input_tensor.shape,kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    x = Activation('relu') (x)
    x = BatchNormalization()(x)
    x = Dropout(0.3) (x)

    x = Conv2D(64, (3, 3), padding='same',name='conv_2',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4) (x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(input_tensor,x)

def train(config, gen, disc_f, disc_h, disc_j, model_en, train_data,layer_model,threshold,layer_to_compute,model_layer_dict):

    # Start training
    # Define optimizers
    disc_optimizer = tf.optimizers.Adam(learning_rate=config.lr_disc,
                                        beta_1=config.beta_1_disc,
                                        beta_2=config.beta_2_disc)

    gen_en_optimizer = tf.optimizers.Adam(learning_rate=config.lr_gen_en,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)


    # Define Logging to Tensorboard
    summary_writer = tf.summary.create_file_writer(f'{config.result_path}/{config.model}_{config.dataset}_{time.strftime("%Y-%m-%d--%H-%M-%S")}')

    fixed_z, fixed_c = get_fixed_random(config, num_to_generate=100)  # fixed_noise is just used for visualization.

    # Define metric
    metric_loss_gen_en = tf.keras.metrics.Mean()
    metric_loss_disc = tf.keras.metrics.Mean()

    # Start training
    epoch_tf = tf.Variable(0, trainable=False, dtype=tf.float32)
    neural_loss = 0
    for epoch in range(config.num_epochs):
                                                   
        logging.info(f'Start epoch {epoch+1} ...')  # logs a message.
        epoch_tf.assign(epoch)
        start_time = time.time()
        if epoch == 100:
                config.train_batch_size == 1
                train_data = train_data[0:5000]
                withcov = True                                  
        train_epoch(train_data, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
                    metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config,withcov,layer_model,threshold,layer_to_compute,model_layer_dict)
        epoch_time = time.time()-start_time

        # Save results
        logging.info(f'Epoch {epoch+1}: Disc_loss: {metric_loss_disc.result()}, Gen_loss: {metric_loss_gen_en.result()}, Time: {epoch_time}')
        with summary_writer.as_default():
            tf.summary.scalar('Generator and Encoder loss',metric_loss_gen_en.result(),step=epoch)
            tf.summary.scalar('Discriminator loss', metric_loss_disc.result(),step=epoch)

        metric_loss_gen_en.reset_states()

        metric_loss_disc.reset_states()
        # Generated images and reconstructed images
        gen_image  = generate_images(gen, fixed_z, fixed_c, config)
        with summary_writer.as_default():
            tf.summary.image('Generated Images', tf.expand_dims(gen_image,axis=0),step=epoch)

def train_epoch(train_data, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer,
                metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config,withcov = False,layer_model,threshold,layer_to_compute,model_layer_dict):
    for image, label in train_data:
        if not config.conditional:
            label = None
        train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer,
                   metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config,withcov,layer_model,threshold,layer_to_compute,model_layer_dict)

@tf.function
def train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
               metric_loss_gen_en, batch_size, cont_dim, config, withcov = False,layer_model,threshold,layer_to_compute,model_layer_dict):
    print('Graph will be traced...')
    
    with tf.device('{}:*'.format(config.device)):
        for _ in range(config.D_G_ratio):
            fake_noise = tf.random.truncated_normal([batch_size, cont_dim])
            with tf.GradientTape(persistent=True) as gen_en_tape, tf.GradientTape() as en_tape:
                fake_img = gen(fake_noise, label, training=True)
                latent_code_real = model_en(image, training=True)
                with tf.GradientTape(persistent=True) as disc_tape:
                    real_f_to_j, real_f_score = disc_f(image, label, training=True)
                    fake_f_to_j, fake_f_score = disc_f(fake_img, label, training=True)
                    real_h_to_j, real_h_score = disc_h(latent_code_real, training=True)
                    fake_h_to_j, fake_h_score = disc_h(fake_noise, training=True)
                    real_j_score = disc_j(real_f_to_j, real_h_to_j, training=True)
                    fake_j_score = disc_j(fake_f_to_j, fake_h_to_j, training=True)
                    ori_dis_loss = disc_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
                    ori_g_e_loss = gen_en_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)                            
                    if withcov :
                        update_coverage(fake_img[0],layer_model,threshold,layer_to_compute,model_layer_dict)
                        neural_loss = 1-neuron_cov(model_layer_dict)[2]
                        
                     d_loss = ori_dis_loss + labmda * neural_loss
                     g_e_loss = ori_g_e_loss + labmda * neural_loss

            grad_disc = disc_tape.gradient(d_loss, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables)

            disc_optimizer.apply_gradients(zip(grad_disc, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables))
            metric_loss_disc.update_state(d_loss)  # upgrade the value in metrics for single step.

        grad_gen_en = gen_en_tape.gradient(g_e_loss, gen.trainable_variables + model_en.trainable_variables)

        gen_en_optimizer.apply_gradients(zip(grad_gen_en, gen.trainable_variables + model_en.trainable_variables))
        metric_loss_gen_en.update_state(g_e_loss)

        del gen_en_tape, en_tape
        del disc_tape




