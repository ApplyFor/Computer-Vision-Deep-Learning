from PyQt5 import QtCore, QtGui, QtWidgets
import sys

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

import os, re

import module

def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    """ 
    Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """
    Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 10, 270, 30))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 60, 270, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(60, 110, 270, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(35, 165, 320, 71))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(25, 25, 270, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(35, 260, 320, 101))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(25, 20, 270, 30))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(25, 60, 270, 30))
        self.pushButton_6.setObjectName("pushButton_6")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(35, 385, 320, 71))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_7.setGeometry(QtCore.QRect(25, 25, 270, 30))
        self.pushButton_7.setObjectName("pushButton_7")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(35, 480, 320, 101))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_8.setGeometry(QtCore.QRect(25, 20, 270, 30))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_9.setGeometry(QtCore.QRect(25, 60, 270, 30))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 40, 300, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 90, 300, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 140, 300, 20))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CvDl 2022 HW2"))
        self.pushButton.setText(_translate("MainWindow", "Load Video"))
        self.pushButton_2.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_3.setText(_translate("MainWindow", "Load Folder"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.pushButton_4.setText(_translate("MainWindow", "1.1 Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Optional Flow"))
        self.pushButton_5.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.pushButton_6.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Perspective Transform"))
        self.pushButton_7.setText(_translate("MainWindow", "3.1 Perspective Transform"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. PCA"))
        self.pushButton_8.setText(_translate("MainWindow", "4.1 Image Reconstruction"))
        self.pushButton_9.setText(_translate("MainWindow", "4.2 Compute the Reconstruction Error"))
        self.label.setText(_translate("MainWindow", "No video loaded"))
        self.label_2.setText(_translate("MainWindow", "No image loaded"))
        self.label_3.setText(_translate("MainWindow", "No folder loaded"))


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.videoname = []
        self.imagename = []
        self.folder = []
        self.images = []
        self.points = []
        self.reconstruct = []

        self.ui.pushButton.clicked.connect(self.load_video)
        self.ui.pushButton_2.clicked.connect(self.load_image)
        self.ui.pushButton_3.clicked.connect(self.load_folder)
        self.ui.pushButton_4.clicked.connect(self.background_subtraction)
        self.ui.pushButton_5.clicked.connect(self.preprocessing)
        self.ui.pushButton_6.clicked.connect(self.video_tracking)
        self.ui.pushButton_7.clicked.connect(self.perspective_transform)
        self.ui.pushButton_8.clicked.connect(self.image_reconstruction)
        self.ui.pushButton_9.clicked.connect(self.reconstruction_error_computation)

    def load_video(self):
        self.videoname, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(), 'All Files (*);;Video Files (*.mp4 *.mov *.wmv *.avi *.mkv)')
        if not self.videoname:
            return
        else:
            self.ui.label.setText(r"{} loaded".format(self.videoname[self.videoname.rfind(r'/')+1:]))

    def load_image(self):
        self.imagename, filetype = QtWidgets.QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(), 'All Files (*);;Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not self.imagename:
            return
        else:
            self.ui.label_2.setText(r"{} loaded".format(self.imagename[self.imagename.rfind(r'/')+1:]))

    def load_folder(self):
        self.images = []
        self.folder = QtWidgets.QFileDialog.getExistingDirectory(self, '選擇資料夾', os.getcwd())
        if not self.folder:
            return
        else:
            self.ui.label_3.setText(r"{} loaded".format(self.folder[self.folder.rfind(r'/')+1:]))

        print("=====LOAD ALL IMAGES IN THE FOLDER=====")
        dir_ = os.listdir(self.folder)
        sort_nicely(dir_)
        for file in dir_:
            print(os.path.join(self.folder, file), end=" ")
            imagetype = [r".*\.png$", r".*\.jpg$", r".*\.jpeg$", r".*\.bmp$"]
            for filetype in imagetype:
                regex = re.compile(filetype)
                filename = regex.findall(file)
                if not filename:
                    continue
                img = cv2.imread(os.path.join(self.folder, filename[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.w, self.h = img.shape[ : -1]
                if img is not None:
                    self.images.append(img)
                    print("LOADED")
        print()

    def background_subtraction(self):
        module.background_subtraction(self.videoname)

    def preprocessing(self):
        self.points = module.preprocessing(self.videoname)

    def video_tracking(self):
        module.video_tracking(self.videoname, self.points)

    def perspective_transform(self):
        module.perspective_transform(self.imagename, self.videoname)

    def image_reconstruction(self):
        self.reconstruct = module.image_reconstruction(self.images, self.w, self.h)

    def reconstruction_error_computation(self):
        module.reconstruction_error_computation(self.images, self.reconstruct)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = GUI()
    MainWindow.show()
    sys.exit(app.exec_())

