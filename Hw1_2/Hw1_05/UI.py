from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.resize(900, 600)
        self.groupBox1 = QtWidgets.QGroupBox(widget)
        self.groupBox1.setGeometry(QtCore.QRect(37, 35, 300, 500))
        self.groupBox1.setObjectName("groupBox1")
        self.pushButton1 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton1.setGeometry(QtCore.QRect(30, 30, 250, 30))
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton2 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton2.setGeometry(QtCore.QRect(30, 100, 250, 30))
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton3 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton3.setGeometry(QtCore.QRect(30, 170, 250, 30))
        self.pushButton3.setObjectName("pushButton3")
        self.pushButton4 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton4.setGeometry(QtCore.QRect(30, 240, 250, 30))
        self.pushButton4.setObjectName("pushButton4")
        self.pushButton5 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton5.setGeometry(QtCore.QRect(30, 310, 250, 30))
        self.pushButton5.setObjectName("pushButton5")
        self.pushButton6 = QtWidgets.QPushButton(self.groupBox1)
        self.pushButton6.setGeometry(QtCore.QRect(30, 380, 250, 30))
        self.pushButton6.setObjectName("pushButton6")
        self.groupBox2 = QtWidgets.QGroupBox(widget)
        self.groupBox2.setGeometry(QtCore.QRect(400, 40, 450, 500))
        self.groupBox2.setTitle("")
        self.groupBox2.setObjectName("groupBox2")
        self.label1 = QtWidgets.QLabel(self.groupBox2)
        self.label1.setGeometry(QtCore.QRect(75, 0, 300, 75))
        self.label1.setObjectName("label1")
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label1.setFont(font)
        self.label2 = QtWidgets.QLabel(self.groupBox2)
        self.label2.setGeometry(QtCore.QRect(25, 60, 400, 400))
        self.label2.setObjectName("label2")

        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "2022 CvDl Hw1"))
        self.groupBox1.setTitle(_translate("widget", "5.VGG19 Test"))
        self.pushButton1.setText(_translate("widget", "Load Image"))
        self.pushButton2.setText(_translate("widget", "1. Show Train Images"))
        self.pushButton3.setText(_translate("widget", "2. Show Model Structure"))
        self.pushButton4.setText(_translate("widget", "3. Show Data Augmentation"))
        self.pushButton5.setText(_translate("widget", "4. Show Accuracy and Loss"))
        self.pushButton6.setText(_translate("widget", "5. Inference"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_widget()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
