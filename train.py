import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import *
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E


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
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)
    model1 = cifar10vgg(input_tensor=input_tensor)
    model1.load_weights("cifar10vgg.h5")
    threshold=0.45
    layer_to_compute = layer_names(model1)
    layer_model = compose_layer_model(model1,layer_to_compute)
    model_layer_dict = set_cov_dict(model1)
    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data,
          layer_model,threshold,layer_to_compute,model_layer_dict
          )
    # Finished
    logging.info('Training finished ;)')
