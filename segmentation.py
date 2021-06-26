from ochumanApi.ochuman import OCHuman
import cv2, os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (15, 15)

import ochumanApi.vis as vistool
from ochumanApi.ochuman import Poly2Mask

import cv2
import numpy as np

IMG_HEIGHT = 512
IMG_WIDTH = 512
epochs = 50
batch_size = 16
ImgDir = "data/images/"

# <Filter>:
#      None(default): load all. each has a bbox. some instances have keypoint and some have mask annotations.
#            images: 5081, instances: 13360
#     'kpt&segm' or 'segm&kpt': only load instances contained both keypoint and mask annotations (and bbox)
#            images: 4731, instances: 8110
#     'kpt|segm' or 'segm|kpt': load instances contained either keypoint or mask annotations (and bbox)
#            images: 5081, instances: 10375
#     'kpt' or 'segm': load instances contained particular kind of annotations (and bbox)
#            images: 5081/4731, instances: 10375/8110
ochuman = OCHuman(AnnoFile='data/annotations/ochuman.json', Filter='segm')
image_ids = ochuman.getImgIds()
print('Total images: %d' % len(image_ids))

def get_segmentation(data):
    img = cv2.imread(os.path.join(ImgDir, data['file_name']))
    height, width = data['height'], data['width']

    colors = [[255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255]]


    for i, anno in enumerate(data['annotations']):
        bbox = anno['bbox']
        kpt = anno['keypoints']
        segm = anno['segms']
        max_iou = anno['max_iou']

        # img = vistool.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
        if segm is not None:
            mask = Poly2Mask(segm)
            img = vistool.draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
        # if kpt is not None:
        #     img = vistool.draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
    return img

def new_mask(real_img, m_img):
    real_img = real_img.reshape(1, -1)[0]
    m_img = m_img.reshape(1, -1)[0]

    new = []

    for i, j in zip(real_img, m_img):
        if i != j:
            new.append(255) # human will white appear because of 255
        else:
            new.append(0) # background will black appear because of 0, set i instead of 0 to do not change backgraound

    new_np = np.array(new)
    new_np = new_np.reshape(512, 512, 3)

    return new_np


def generator_images(batch_size=1, ind=0):
    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            data = ochuman.loadImgs(imgIds=[image_ids[ind]])[0]

            file_name = data['file_name']

            img = cv2.imread(ImgDir + '/' + file_name)

            y = get_segmentation(data)

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            y = cv2.resize(y, (IMG_WIDTH, IMG_HEIGHT))

            new = new_mask(img, y)

            img = img / 255.
            y = new / 255.

            x_batch.append(img)
            y_batch.append(y)

        x_batch = np.array(x_batch)

        y_batch = {'seg': np.array(y_batch)
                   }

        yield x_batch, y_batch


# for i in range(4731):
#     for x, y in generator_images(1, i):
#         break
#
#     base_dir_custom = "custom_dataset_human_black_background/"
#     try:
#         os.makedirs(f'{base_dir_custom}')
#     except:
#         pass
#     try:
#         os.makedirs(f'{base_dir_custom}features/')
#     except:
#         pass
#     try:
#         os.makedirs(f'{base_dir_custom}labels/')
#     except:
#         pass
#
#     x_name = f"{base_dir_custom}features/{i}_x.jpg"
#     y_name = f"{base_dir_custom}labels/{i}_y.jpg"
#     cv2.imwrite(x_name, x[0] * 255.)
#     cv2.imwrite(y_name, y['seg'][0] * 255.)


features = os.listdir("custom_dataset_human_black_background/features/")
labels = os.listdir("custom_dataset_human_black_background/labels/")

print(len(features), len(labels))

X = features
y = labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.15, random_state=1)

print(len(X_train), len(X_val), len(X_test))


def keras_generator_train_val_test(batch_size, choice="train"):
    if choice == "train":
        X = X_train
        y = y_train
    elif choice == "val":
        X = X_val
        y = y_val
    elif choice == "test":
        X = X_test
        y = y_test
    else:
        print("Invalid Option")
        return False

    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            x_rand = random.choice(X)
            y_rand = x_rand[:-5] + "y.jpg"

            x_path = f"{ImgDir}features/{x_rand}"
            y_path = f"{ImgDir}labels/{y_rand}"

            x = cv2.imread(x_path)
            y = cv2.imread(y_path)

            x = x / 255.
            y = y / 255.

            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch)
        # y_batch = np.array(y_batch)

        y_batch = {'seg': np.array(y_batch),
                   #    'cls': np.array(classification_list)
                   }

        yield x_batch, y_batch


def get_model():
    in1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(in1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

    up1 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up2 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    segmentation = Conv2D(3, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])

    losses = {'seg': 'binary_crossentropy'
              }

    metrics = {'seg': ['acc']
               }
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    return model


import datetime


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs={}):

        res_dir = "intermediate_results_purple_background"

        try:
            os.makedirs(res_dir)
        except:
            print(f"{res_dir} directory already exist")

        print('Training: epoch {} begins at {}'.format(epoch, datetime.datetime.now().time()))

    def on_epoch_end(self, epoch, logs=None):
        res_dir = "intermediate_results_purple_background/"
        print('Training: epoch {} ends at {}'.format(epoch, datetime.datetime.now().time()))

        for x_test, y_test in keras_generator_train_val_test(batch_size, choice="test"):
            break
        p = np.reshape(x_test[0], (1, 512, 512, 3))
        prediction = self.model.predict(p)

        x_img = f"{res_dir}{epoch}_X_input.jpg"
        y_img = f"{res_dir}{epoch}_Y_truth.jpg"
        predicted_img = f"{res_dir}{epoch}_Y_predicted.jpg"

        cv2.imwrite(x_img, x_test[0] * 255.)
        cv2.imwrite(y_img, y_test['seg'][0] * 255.)
        cv2.imwrite(predicted_img, prediction[0] * 255.)


model_name = "models/" + "Unet_purple_background.h5"

modelcheckpoint = ModelCheckpoint(model_name,
                                  monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True)

lr_callback = ReduceLROnPlateau(min_lr=0.000001)

callback_list = [modelcheckpoint, lr_callback, MyCustomCallback()]

history = get_model().fit_generator(
    keras_generator_train_val_test(batch_size, choice="train"),
    validation_data=keras_generator_train_val_test(batch_size, choice="val"),
    validation_steps=100,
    steps_per_epoch=100,
    epochs=epochs,
    verbose=1,
    shuffle=True,
    callbacks=callback_list,
)
