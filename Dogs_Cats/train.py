# utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from tensorflow import keras
import sys
import cv2


print(tf.__version__)
print(sys.version_info)
for module in mpl, pd, np, keras:
    print(module.__name__, module.__version__)

# set path
train_dir = './train'
valid_dir = './validation'
test_dir = './test'

# set para
height = 128
width = 128
channel = 3
batch_size = 64
valid_batch_size = 64
num_classes = 2
epochs = 100


# train model
def trainModel(model, train_generator, valid_generator, callbacks):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data = valid_generator,
        callbacks = callbacks
    )
    return history

# show the changes of loss and accuracy during training
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


# use the model after finish training to classify images and save them into two different folders
def predictModel(model, output_model_file):

    # load the weights of model
    model.load_weights(output_model_file)

    os.makedirs('./save', exist_ok=True)
    os.makedirs('./save/cat', exist_ok=True)
    os.makedirs('./save/dog', exist_ok=True)

    test_dir = './test/'  # 1-12500.jpg
    for i in range(1, 12500):
        img_name = test_dir + '{}.jpg'.format(i)
        img = cv2.imread(img_name)
        img = cv2.resize(img, (width, height))
        img_arr = img / 255.0
        img_arr = img_arr.reshape((1, width, height, 3))
        pre = model.predict(img_arr)
        if pre[0][0] > pre[0][1]:
            cv2.imwrite('./save/cat/' + '{}.jpg'.format(i), img)
            print(img_name, ' is classified as Cat.')
        else:
            cv2.imwrite('./save/dog/' + '{}.jpg'.format(i), img)
            print(img_name, ' is classified as Dog.')


if __name__ ==  '__main__':

    print('Start importing data...')

    # import the data and make a augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(width, height),
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        class_mode='categorical'
    )

    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(width, height),
        batch_size=valid_batch_size,
        seed=7,
        shuffle=False,
        class_mode="categorical"
    )


    train_num = train_generator.samples
    valid_num = valid_generator.samples
    # print(train_num, valid_num)

    print('Building model...')

    # Build model
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3,
                            padding='same', activation='relu',
                            input_shape=[width, height, channel]),
        keras.layers.Conv2D(filters=32, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Conv2D(filters=128, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.Conv2D(filters=128, kernel_size=3,
                            padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=2),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    # set path of saving model
    logdir = './graph_def_and_weights'
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    output_model_file = os.path.join(logdir,
                                     "catDog_weights.h5")


    print('Start training ...')
    # start training

    mode = input('Select mode: 1.Train 2.Predict\nInput number: ')
    if mode == '1':
        callbacks = [
            keras.callbacks.TensorBoard(logdir),
            keras.callbacks.ModelCheckpoint(output_model_file,
                                            save_best_only=True,
                                            save_weights_only=True),
            keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
        ]
        history = trainModel(model, train_generator, valid_generator, callbacks)
        plot_learning_curves(history, 'accuracy', epochs, 0, 1)
        plot_learning_curves(history, 'loss', epochs, 0, 5)
    elif mode == '2':
        # Only run this mode if you have already finished training your model and saved it.
        predictModel(model, output_model_file)
    else:
        print('Please input the correct number.')

    print('Finish! Exit.')
