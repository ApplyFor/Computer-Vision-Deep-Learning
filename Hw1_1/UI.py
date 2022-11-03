from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_widget(object):
    def setupUi(self, widget):
        widget.setObjectName("widget")
        widget.resize(1000, 500)
        self.groupBox = QtWidgets.QGroupBox(widget)
        self.groupBox.setGeometry(QtCore.QRect(40, 30, 200, 400))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 70, 160, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 170, 160, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 280, 160, 40))
        self.pushButton_3.setObjectName("pushButton_3")
        self.groupBox_2 = QtWidgets.QGroupBox(widget)
        self.groupBox_2.setGeometry(QtCore.QRect(280, 30, 200, 400))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setGeometry(QtCore.QRect(20, 30, 160, 40))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 90, 160, 40))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 150, 180, 120))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_7.setGeometry(QtCore.QRect(10, 70, 160, 40))
        self.pushButton_7.setObjectName("pushButton_7")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(50, 30, 90, 25))  #設定相對groupBox_3座標位置(50, 30) 寬90x高25
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("1")
        self.comboBox.addItem("2")
        self.comboBox.addItem("3")
        self.comboBox.addItem("4")
        self.comboBox.addItem("5")
        self.comboBox.addItem("6")
        self.comboBox.addItem("7")
        self.comboBox.addItem("8")
        self.comboBox.addItem("9")
        self.comboBox.addItem("10")
        self.comboBox.addItem("11")
        self.comboBox.addItem("12")
        self.comboBox.addItem("13")
        self.comboBox.addItem("14")
        self.comboBox.addItem("15")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_8.setGeometry(QtCore.QRect(20, 280, 160, 40))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_9.setGeometry(QtCore.QRect(20, 340, 160, 40))
        self.pushButton_9.setObjectName("pushButton_9")
        self.groupBox_4 = QtWidgets.QGroupBox(widget)
        self.groupBox_4.setGeometry(QtCore.QRect(520, 30, 200, 400))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_10.setGeometry(QtCore.QRect(10, 180, 180, 40))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_11.setGeometry(QtCore.QRect(10, 280, 180, 40))
        self.pushButton_11.setObjectName("pushButton_11")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_4)
        self.textEdit.setGeometry(QtCore.QRect(10, 80, 180, 40))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setFontPointSize(12)
        self.groupBox_5 = QtWidgets.QGroupBox(widget)
        self.groupBox_5.setGeometry(QtCore.QRect(760, 30, 200, 400))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_12.setGeometry(QtCore.QRect(10, 180, 180, 40))
        self.pushButton_12.setObjectName("pushButton_12")

        self.retranslateUi(widget)
        QtCore.QMetaObject.connectSlotsByName(widget)

    def retranslateUi(self, widget):
        _translate = QtCore.QCoreApplication.translate
        widget.setWindowTitle(_translate("widget", "2022 CvDl Hw1"))    #視窗名稱
        self.groupBox.setTitle(_translate("widget", "Load Image"))
        self.pushButton.setText(_translate("widget", "Load Folder"))
        self.pushButton_2.setText(_translate("widget", "Load Image_L"))
        self.pushButton_3.setText(_translate("widget", "Load Image_R"))
        self.groupBox_2.setTitle(_translate("widget", "1. Calibration"))
        self.pushButton_4.setText(_translate("widget", "1.1 Find Corners"))
        self.pushButton_5.setText(_translate("widget", "1.2 Find Intrinsic"))
        self.groupBox_3.setTitle(_translate("widget", "1.3 Find Extrinsic"))
        self.pushButton_7.setText(_translate("widget", "1.3 Find Extrinsic"))
        self.comboBox.setItemText(0, _translate("widget", "1"))
        self.comboBox.setItemText(1, _translate("widget", "2"))
        self.comboBox.setItemText(2, _translate("widget", "3"))
        self.comboBox.setItemText(3, _translate("widget", "4"))
        self.comboBox.setItemText(4, _translate("widget", "5"))
        self.comboBox.setItemText(5, _translate("widget", "6"))
        self.comboBox.setItemText(6, _translate("widget", "7"))
        self.comboBox.setItemText(7, _translate("widget", "8"))
        self.comboBox.setItemText(8, _translate("widget", "9"))
        self.comboBox.setItemText(9, _translate("widget", "10"))
        self.comboBox.setItemText(10, _translate("widget", "11"))
        self.comboBox.setItemText(11, _translate("widget", "12"))
        self.comboBox.setItemText(12, _translate("widget", "13"))
        self.comboBox.setItemText(13, _translate("widget", "14"))
        self.comboBox.setItemText(14, _translate("widget", "15"))
        self.pushButton_8.setText(_translate("widget", "1.4 Find Distortion"))
        self.pushButton_9.setText(_translate("widget", "1.5 Show Result"))
        self.groupBox_4.setTitle(_translate("widget", "2. Augmented Reality"))
        self.pushButton_10.setText(_translate("widget", "2.1 Show Words on Board"))
        self.pushButton_11.setText(_translate("widget", "2.2 Show Words Vertically"))
        self.groupBox_5.setTitle(_translate("widget", "3. Stereo Disparity Map"))
        self.pushButton_12.setText(_translate("widget", "3.1 Stereo Disparity Map"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_widget()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
