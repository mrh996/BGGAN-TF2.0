import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import *
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E
import random
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

def set_up_train(config):
    # Setup tensorflow
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data, _ = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data, config)

    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model_generator = BIGBIGAN_G(config, weight_init)
    model_discriminator_f = BIGBIGAN_D_F(config, weight_init)
    model_discriminator_h = BIGBIGAN_D_H(config, weight_init)
    model_discriminator_j = BIGBIGAN_D_J(config, weight_init)
    model_encoder = BIGBIGAN_E(config, weight_init)

    # train
    logging.info('Start training...')

    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data)
    # Finished
    logging.info('Training finished ;)')
