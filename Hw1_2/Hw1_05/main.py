import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

from UI import Ui_widget
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from keras.layers import VersionAwareLayers
from keras.engine import training

import torch
from torchsummary import summary
from torchvision import models

from PIL import Image
from torchvision import transforms

from tensorflow.keras.preprocessing import image

import random

def show_train_images():
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    plt.figure()
    for row in range(3): #0-2
        for col in range(3):
            rand = random.randrange(0, 50000) #0-49999
            #matlab initial index = 1
            plt.subplot(3, 3, 1+3*row+col)
            plt.title('{}'.format(label[y_train[rand][0]])) #str.format()
            plt.axis('off')
            plt.imshow(x_train[rand])

    plt.show()

def vgg19(input_shape, classes):
    '''
    layers = VersionAwareLayers()
    img_input = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv4")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv4")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv4")(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    
     # Classification block
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)
    x = layers.Dense(classes, activation="softmax", name="predictions")(x)
    
    inputs = img_input
    '''
    
    vgg = VGG19(include_top = False, weights = 'imagenet', input_shape = input_shape)

    x = tf.keras.layers.Flatten(name="flatten")(vgg.output)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dense(classes, activation="softmax", name="predictions")(x)

    inputs = vgg.input
    
    vgg19 = training.Model(inputs, x, name="vgg19")
    return vgg19

    '''
    x = models.vgg19()
    x.classifier._modules['6'] = torch.nn.Linear(4096, 10)
    return x
    '''

def show_model_structure():
    global model
    #model = vgg19((32, 32, 3), 10)
    model = tf.keras.models.load_model('model.pth')
    model.summary()
    
    '''
    model = torch.load('model.pth')
    #print(model)
    summary(model, (3, 32, 32))
    '''

def show_accuracy_and_loss():
    global model
    # 数据，切分为训练和测试集。
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # 将类向量转换为二进制类矩阵。(onehot)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    #normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #verbose 日誌顯示
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test), shuffle=True)

    model.save('model.pth')

    plt.figure()
    # 绘制训练 & 验证的准确率值
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('%')
    plt.legend(['Training', 'Testing'], loc='lower right')

    # 绘制训练的损失值
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')

    #plt.savefig('plot.png')
    plt.show()
    

class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_widget()
        self.ui.setupUi(self)

        self.ui.pushButton1.clicked.connect(self.load_image)
        self.ui.pushButton2.clicked.connect(show_train_images)
        self.ui.pushButton3.clicked.connect(show_model_structure)
        self.ui.pushButton4.clicked.connect(self.show_data_augmentation)
        self.ui.pushButton5.clicked.connect(self.show_accuracy_and_loss)
        self.ui.pushButton6.clicked.connect(self.inference)
    
    def load_image(self):
        self.filename, self.filetype = QtWidgets.QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(), 'All Files (*);;Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not self.filename:
            print('No file selected')
            return

        self.ui.label1.setText("")
        self.mypixmap = QtGui.QPixmap(self.filename)
        
        if self.mypixmap.isNull():
            print('Load image failed')
            return
        self.mypixmap = self.mypixmap.scaled(self.ui.label2.size().width(), self.ui.label2.size().height(), QtCore.Qt.KeepAspectRatio)
        self.ui.label2.setPixmap(self.mypixmap)

    def show_data_augmentation(self):
        if not self.filename:
            print('No file loaded')
            return
        image = Image.open(self.filename).convert('RGB')

        transforms1 = transforms.Compose([
            transforms.Resize((1000, 1000)),
            transforms.RandomRotation(90),
        ])
        image1 = transforms1(image)

        transforms2 = transforms.Compose([
            transforms.Resize((1000, 1000)),
            transforms.RandomResizedCrop((1000, 1000))
        ])
        image2 = transforms2(image)

        transforms3 = transforms.Compose([
            transforms.Resize((1000, 1000)),
            transforms.ColorJitter(brightness=.5, hue=.3)
        ])
        image3 = transforms3(image)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('RandomRotation()')
        plt.axis('off')
        plt.imshow(image1)
        plt.subplot(1, 3, 2)
        plt.title('RandomResizedCrop()')
        plt.axis('off')
        plt.imshow(image2)
        plt.subplot(1, 3, 3)
        plt.title('ColorJitter()')
        plt.axis('off')
        plt.imshow(image3)
        plt.show()

    def show_accuracy_and_loss(self):
        self.ui.label1.setText("")
        self.mypixmap = QtGui.QPixmap('plot.png')
        if self.mypixmap.isNull():
            print('Load image failed')
            return
        self.mypixmap = self.mypixmap.scaled(self.ui.label2.size().width(), self.ui.label2.size().height(), QtCore.Qt.KeepAspectRatio)
        self.ui.label2.setPixmap(self.mypixmap)

    def inference(self):
        global model
        label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if not self.filename:
            print('No file loaded')
            return
        test_image = image.load_img(self.filename, target_size=(32, 32))
        array = image.img_to_array(test_image)
        x = np.expand_dims(array, axis =0) #(1,w,h)
        x/= 255

        model = tf.keras.models.load_model('model.pth')

        predictions = model.predict(x)
        #print(predictions)
        index = np.argmax(predictions)
        pl = label[index] #prediction label
        #print(pl)

        confidence = predictions[0][index]#confidence
        #print(confidence)

        self.ui.label1.setText("Confidence = {:.2f}\nPrediction Label = {}".format(confidence, pl))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = GUI()
    widget.show()
    sys.exit(app.exec_())
