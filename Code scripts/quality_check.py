# modified code from the book: Rowel Atienza. „Advanced deep learning with TensorFlow 2 and Keras: apply DL,GANs, VAEs, deep RL, unsupervised learning, object detection and segmentation, andmore”.
'''Example of autoencoder model on MNIST dataset
This autoencoder has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from os import listdir

from sklearn.svm import OneClassSVM

import os

def img_loader(path, img_size=(128, 128)):
    temp = []
    for file in listdir(path):
        temp.append(img_to_array((load_img(path + "\\" + file, color_mode='grayscale', target_size=img_size))))
    return np.array(temp)

main_path = r'D:\Cyfronet\Resised real'
temp = ['Normal']
for dir in temp:

    # test_path = os.getcwd() + r"\Test"
    test_path = main_path + '\\' + dir + '\\' + r"Test"
    # train_path = os.getcwd() + r"\Train"
    train_path = main_path + '\\' + dir + '\\' + r"Train"
    print(train_path)
    print(test_path)
    if dir == 'Normal':

        x_train_original_images = img_loader(train_path, (152, 256)) #resizing images, so that the encoder will work properly
        x_test_original_images = img_loader(test_path, (152, 256))

    elif dir == 'Cutted_pictures':
        x_train_original_images = img_loader(train_path,
                                             (128, 128))  # resizing images, so that the encoder will work properly
        x_test_original_images = img_loader(test_path, (128, 128))

    elif dir == 'Full_pictures':
        x_train_original_images = img_loader(train_path,
                                             (256, 256))  # resizing images, so that the encoder will work properly
        x_test_original_images = img_loader(test_path, (256, 256))

    image_size_x = x_train_original_images.shape[2]
    image_size_y = x_train_original_images.shape[1]

    x_train_original_images = np.reshape(x_train_original_images, [-1, image_size_y, image_size_x, 1])
    print(x_train_original_images.shape)
    x_test_original_images = np.reshape(x_test_original_images, [-1, image_size_y, image_size_x, 1])
    x_train_original_images = x_train_original_images.astype('float32') / 255
    x_test_original_images = x_test_original_images.astype('float32') / 255

    # network parameters
    input_shape = (image_size_y, image_size_x, 1)
    batch_size = 4
    kernel_size = 3
    latent_dim = 32 # best
    # encoder/decoder number of CNN layers and filters per layer
    layer_filters = [32, 64, 128]

    # build the autoencoder model
    # first build the encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # stack of Conv2D(32)-Conv2D(64)
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    # so we don't do hand computation
    # the input to the decoder's first
    # Conv2DTranspose will have this shape
    # shape is (7, 7, 64) which is processed by
    # the decoder back to (28, 28, 1)
    shape = K.int_shape(x)

    # generate latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # instantiate encoder model
    encoder = Model(inputs,
                    latent,
                    name='encoder')
    encoder.summary()


    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    # from vector to suitable shape for transposed conv
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)

    # reconstruct the input
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='decoder.png', show_shapes=True)


    # autoencoder = encoder + decoder
    # instantiate autoencoder model

    autoencoder = Model(inputs,
                        decoder(encoder(inputs)),
                        name='autoencoder')
    autoencoder.summary()
    # plot_model(autoencoder,
    #            to_file='autoencoder.png',
    #            show_shapes=True)

    # Mean Square Error (MSE) loss function, Adam optimizer
    autoencoder.compile(loss='mse', optimizer='adam')
    # checkpoint_dir = r'./saved_weights/' + dir + r'/weights'
    checkpoint_dir = r'./saved_weights/' + dir
    # train the autoencoder
    print(checkpoint_dir)
    if tf.train.latest_checkpoint(checkpoint_dir) != None:

        latest = tf.train.latest_checkpoint(checkpoint_dir)
        autoencoder.load_weights(latest)
        print('Loaded ' + dir + 'weights')
    else:
        print('Not loaded')
        autoencoder.fit(x_train_original_images,
                        x_train_original_images,
                        validation_data=(x_test_original_images, x_test_original_images),
                        epochs=25,
                        batch_size=batch_size)
        print(checkpoint_dir)
        autoencoder.save_weights(checkpoint_dir + r'/' + 'weights')

    # predict the autoencoder output from test data
    x_decoded = autoencoder.predict(x_test_original_images)

    # display the 1st 8 test input and decoded images
    imgs = np.concatenate([x_test_original_images[:8], x_decoded[:8]])
    imgs = imgs.reshape((2, 8, image_size_y, image_size_x))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('input_and_decoded.png')
    plt.show()


    encoder = Model(autoencoder.input, autoencoder.layers[-2].output) # taking the encoder layers from the autoencoder

    # decoder_input = Input(shape=(encoding_dim,))
    # decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

    encoder.summary()
    # decoder.summary()


    # generated_path =os.getcwd() + r"\Cyfronet_Samples_seperate" # path to genrated files
    batches = ['4 batches', '8 batches', '12 batches', '16 batches']

#-----------------------------------------------------------------------------------------------------------------------
# The encoding latent vectors part
#-----------------------------------------------------------------------------------------------------------------------
    
    for b in batches:
        gen_img_path = r'D:\cyfronet images\Seperate' + '\\' + b + '\\' + dir
        for epochs in listdir(gen_img_path):

            generated_path = gen_img_path + '\\' + epochs + '\\' + 'Generated'
            if dir == 'Normal':
                print('Loading: ', generated_path)
                gen_images = img_loader(generated_path, (152, 256))
            elif dir == 'Full_pictures':
                print('Loading: ', generated_path)
                gen_images = img_loader(generated_path, (256, 256))
            elif dir == 'Cutted_pictures':
                print('Loading: ', generated_path)
                gen_images = img_loader(generated_path, (128, 128))
            result_generated_test_x = encoder.predict(gen_images) # generating the latent vector for generated imges from a GAN
            result_real_train_x = encoder.predict(x_train_original_images)
            # print("the Result train: ", result_real_train_x)
            result_real_test_x = encoder.predict(x_test_original_images) # generating the latent vector for real images

            x_test_original_images_result = np.concatenate((result_generated_test_x[:200], result_real_test_x), axis=0) #taking a couple of images for testing

            result_generated_test_y = np.ones(2085)*0
            result_real_test_y = np.ones(42)

            y_test_result = np.concatenate((result_generated_test_y[:200], result_real_test_y), axis=0)


#-----------------------------------------------------------------------------------------------------------------------
            # the predicting SVM part
#-----------------------------------------------------------------------------------------------------------------------
            one_models = r'./scikit_models/' + dir + '.joblib'
            if os.path.isfile(one_models):
                print("Existing ONEClass model")
                print('Loading the model...')
                a_file = open(one_models, 'rb')
                classifier = pickle.load(a_file)
                a_file.close()
            else:
                print('No ONEClass model present')
                print('Creating one...')
                classifier = OneClassSVM(kernel='sigmoid', gamma = 'auto', degree=5)
                print('Training...')
                classifier.fit(result_real_train_x)
                a_file = open(one_models, 'wb')
                pickle.dump(classifier, a_file)
                print(a_file, '--- Has been created!')
                a_file.close()
                print('Created!!!')
            print(result_generated_test_x.shape)
            print(result_real_train_x.shape)

            gen_pred_result = classifier.predict(result_generated_test_x)
            real_pred_result = classifier.predict(result_real_test_x)
            real_pred_train_result = classifier.predict(result_real_train_x)

            print("Generated predict: ", gen_pred_result, sum(gen_pred_result))
            print("Real predicct:", real_pred_result)
            print("Real train predicct:", real_pred_train_result)

            score_gen = classifier.score_samples(result_generated_test_x[-200:])
            score_real = classifier.score_samples(result_real_test_x)


            print('Score of samples gen: \n', np.mean(classifier.score_samples(result_generated_test_x[-200:])))

            print('Score of samples train: \n', np.mean(classifier.score_samples(result_real_train_x)))
            print('Score of samples test: \n', np.mean(classifier.score_samples(result_real_test_x)))

            file_name = r'oneclass_results' + '\\' + b + '_' + dir + '_' + epochs + '_' + 'predict_gen.pkl'
            a_file = open(file_name, "wb")
            pickle.dump(gen_pred_result[:1000], a_file)
            print(a_file, '--- Han been created!')
            a_file.close()

            file_name = r'oneclass_results' + '\\' + b + '_' + dir + '_' + epochs + '_' + 'score_gen.pkl'
            a_file = open(file_name, "wb")
            pickle.dump(score_gen[:1000], a_file)
            print(a_file, '--- Han been created!')
            a_file.close()

            file_score = r'oneclass_results' + '\\' + b + '_' + dir + '_' + epochs + '_' + 'score_test.pkl'
            file_test = r'oneclass_results' + '\\' + b + '_' + dir + '_' + epochs + '_' + 'predict_test.pkl'

            if os.path.isfile(file_score) and os.path.isfile(file_test):
                continue
            else:
                print("Files ", file_score, 'and ', file_test, 'are missing')
                print('Creating them...')
                file_name = r'oneclass_results' + '\\' + b + '_' + dir + '_' + 'score_test.pkl'
                a_file = open(file_name, "wb")
                pickle.dump(score_real[:1000], a_file)
                print(a_file, '--- Han been created!')
                a_file.close()

                file_name = r'oneclass_results' + '\\' + b + '_' + dir + '_' + 'predict_test.pkl'
                a_file = open(file_name, "wb")
                pickle.dump(real_pred_result[:1000], a_file)
                print(a_file, '--- Han been created!')
                a_file.close()

print("The whole big chunky process has finished :D")