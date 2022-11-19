import os
import pandas as pd
import numpy as np
import shutil
import cv2
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_addons.metrics import F1Score
from keras.utils import np_utils
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import json



def grouping_imag_files(train_df, no_of_labels):

    # create directories in train
    data_dir_name = "train_test_data"
    train_data_dir_name = "train"
    for i in range(no_of_labels):
        folder_path = os.path.join(data_dir_name, train_data_dir_name, str(i))
        os.makedirs(folder_path,exist_ok=True)

    # move the files to the labelled directory
    for index, row in train_df.iterrows():
        cur_label = row.values[0]
        cur_data_dir_name, cur_train_dir_name, cur_file_path = row.values[4].split("/")
        src_path = os.path.join(cur_data_dir_name, cur_train_dir_name, cur_file_path)
        dst_path = os.path.join(cur_data_dir_name, cur_train_dir_name, str(cur_label))
        if not os.path.exists(dst_path):
            shutil.move(src_path,dst_path)

def createTrainingData(train_path, CATEGORIES, IMG_SIZE):
    training = []
    for category in CATEGORIES:
        path = os.path.join(train_path, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training.append([new_array, class_num])
    return training


def createTest(test_path, IMG_SIZE):
    testing = []
    for img in os.listdir(test_path):
        img_array = cv2.imread(os.path.join(test_path,img))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        testing.append(new_array)

    return testing

def image_transformation(IMG_SIZE, CHANNELS, data):

    X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
    X = X.astype('float32')
    X /= 255

    return X


def split_data(training, IMG_SIZE, NUM_OF_LABELS, CHANNELS, TEST_SIZE, SEED):
    X =[]
    y =[]

    for features, label in training:
        X.append(features)
        y.append(label)
    
    image_transformation(IMG_SIZE, CHANNELS, X)
    Y = np_utils.to_categorical(y, NUM_OF_LABELS)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = SEED, stratify = Y)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (X_train, X_test, y_train, y_test)

def image_augmentation(IMG_SIZE, CHANNELS):

    data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical",
                        input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        tf.keras.layers.RandomRotation(0.4),
        tf.keras.layers.RandomZoom(0.1)
    ]
    )

    resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.Rescaling(1./255)
    ])

    return data_augmentation, resize_and_rescale

if __name__ == '__main__':

    # Import csv files
    DIR_NAME = "train_test_data"
    TRAIN_DIR_NAME = "train"
    TEST_DIR_NAME = "test"
    train_path = os.path.join(DIR_NAME, TRAIN_DIR_NAME)
    test_path = os.path.join(DIR_NAME, TEST_DIR_NAME)
    traindf = pd.read_csv('train.csv')
    testdf = pd.read_csv('test.csv')
    print(traindf.head())

    NUM_OF_LABELS = len(np.unique(traindf['label']))
    grouping_imag_files(traindf, NUM_OF_LABELS)


    SEED = 4
    CHANNELS = 3
    CATEGORIES = list(map(str, np.unique(traindf["label"])))
    IMG_SIZE = 256
    TEST_SIZE = 0.2

    training = createTrainingData(train_path, CATEGORIES, IMG_SIZE)
    random.shuffle(training)
    
    
    X_train, X_test, y_train, y_test = split_data(training, IMG_SIZE, NUM_OF_LABELS, CHANNELS, TEST_SIZE, SEED)


    data_augmentation, resize_and_rescale = image_augmentation(IMG_SIZE, CHANNELS)


    BATCH_SIZE = 32
    NB_EPOCHS = 100
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)


    model = tf.keras.Sequential([
        # data_augmentation,
        # resize_and_rescale, 
        # Layer 1
        Conv2D(filters = 16, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            input_shape = (IMG_SIZE, IMG_SIZE, 3),
            activation="relu"),
        # Layer 2
        Conv2D(filters = 32, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            activation="relu"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        # Layer 3
        Conv2D(filters = 64, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            activation="relu"),
        Dropout(0.1),
        # Layer 4
        Conv2D(filters = 128, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            activation="relu"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        Flatten(),
        # Output Layer
        Dense(units = 512, activation = "relu"),
        Dense(units = 3, activation = "softmax"),
        ])

    # training
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[F1Score(num_classes= NUM_OF_LABELS, average = "macro")])
    model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = NB_EPOCHS, verbose = 1, validation_data = (X_test, y_test))


    testing = createTest(test_path, IMG_SIZE)
    test = image_transformation(IMG_SIZE, CHANNELS, testing)

    result = dict()
    i = 7000
    predictions = model.predict(test)
    for prediction in predictions:
        result[str(i)] = int(np.argmax(prediction))
        i += 1
    
    z = dict({"target":result})
    output = open("predictions.json", "w")
    json.dump(z, output)
    output.close()