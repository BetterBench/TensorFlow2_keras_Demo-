import os

import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 载入mnist数据集
dataset, metadata = tfds.load("mnist", as_supervised=True, with_info=True)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_test, y_test = x_test / 255.0, y_test / 255.0

def savePic(pic_num):
    '''创建文件路径并保存图片'''
    save_dir = 'test_pic/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    for i in range(pic_num):
        image_array = x_train[i, :]
        image_array = image_array.reshape(28, 28)
        filename = save_dir + 'mnist_train_%d.jpg' % i
        scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)


def img2vec(path):
    '''
    jpg转换为向量
    Args:
        path: the path of your pic
    '''
    try:
        im = Image.open(path).convert('L')
        im = im.resize((28, 28))
        tmp = np.array(im)
        vec = tmp.ravel()
        return vec
    except:
        print("图片不存在")


if __name__ == '__main__':

    # 保存20张图片用于测试
    # savePic(20)

    print("请输入图片路径:")
    # path = input()
    path = './test_pic/mnist_train_0.jpg'
    im = Image.open(path)
    plt.imshow(im)
    plt.show()
    pic = np.array(img2vec(path))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个保存模型权重的回调
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    verbose=1)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
            epochs=100,
            callbacks=[cp_callback])

    print("\n识别结果:\n{}".format(np.argmax(model.predict(pic.reshape(1, 28, 28)))))
    print()
