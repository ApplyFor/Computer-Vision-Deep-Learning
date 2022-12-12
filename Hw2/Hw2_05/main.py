from PyQt5 import QtCore, QtGui, QtWidgets
import sys

import cv2

import os

import train

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 30, 250, 500))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(25, 40, 200, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(25, 110, 200, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(25, 180, 200, 40))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(25, 250, 200, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(25, 320, 200, 40))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_6.setGeometry(QtCore.QRect(25, 390, 200, 40))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(400, 50, 280, 400))
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(500, 460, 120, 15))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 Opencvdl Hw2"))
        self.groupBox.setTitle(_translate("MainWindow", "5.ResNet50"))
        self.pushButton.setText(_translate("MainWindow", "Load Image"))
        self.pushButton_2.setText(_translate("MainWindow", "1. Show Images"))
        self.pushButton_3.setText(_translate("MainWindow", "2. Show Distribution"))
        self.pushButton_4.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.pushButton_5.setText(_translate("MainWindow", "4. Show Comparison"))
        self.pushButton_6.setText(_translate("MainWindow", "5. Inference"))

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.filename = []
        self.model = None
        self.device = None

        self.ui.pushButton.clicked.connect(self.load_image)
        self.ui.pushButton_2.clicked.connect(self.show_images)
        self.ui.pushButton_3.clicked.connect(train.show_distribution)
        self.ui.pushButton_4.clicked.connect(self.show_model_structure)
        self.ui.pushButton_5.clicked.connect(self.show_comparison)
        self.ui.pushButton_6.clicked.connect(self.inference)

    def load_image(self):
        self.filename, self.filetype = QtWidgets.QFileDialog.getOpenFileName(self, '開啟檔案', os.getcwd(), 'All Files (*);;Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not self.filename:
            print('No file selected')
            return

        self.ui.label_2.setText("")
        self.mypixmap = QtGui.QPixmap(self.filename)
        
        if self.mypixmap.isNull():
            print('Load image failed')
            return
        self.mypixmap = self.mypixmap.scaled(self.ui.label.size().width(), self.ui.label.size().height(), QtCore.Qt.KeepAspectRatio)
        self.ui.label.setPixmap(self.mypixmap)

    def show_images(self):
        self.train_dataloader, self.validation_dataloader = train.show_images()

    def show_model_structure(self):
        self.model = train.show_model_structure()

    def show_comparison(self):
        train.show_comparison(self.model, self.train_dataloader, self.validation_dataloader)

    def inference(self):
        if not self.filename:
            print('No image loaded')
            return

        img = cv2.imread(self.filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = train.inference(img)
        self.ui.label_2.setText("Prediction: {}".format(label))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = GUI()
    MainWindow.show()
    sys.exit(app.exec_())