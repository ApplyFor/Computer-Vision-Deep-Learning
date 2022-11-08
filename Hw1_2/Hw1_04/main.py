import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

image_1 = []
image_2 = []
def load_image_1():
    global image_1
    image_1 = cv2.imread(r"Dataset_CvDl_Hw1_2/Q4_images/traffics.png")

def load_image_2():
    global image_2
    image_2 = cv2.imread(r"Dataset_CvDl_Hw1_2/Q4_images/ambulance.png")

def keypoints():
    image_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    image_keypoints = cv2.drawKeypoints(image_gray, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.namedWindow("traffics.png", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("traffics.png", (int(image_1.shape[1]*1.5), int(image_1.shape[0]*1.5)))
    cv2.moveWindow("traffics.png", 150, 150)
    cv2.imshow("traffics.png", image_1)

    cv2.namedWindow("Figure 1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Figure 1", (int(image_keypoints.shape[1]*1.5), int(image_keypoints.shape[0]*1.5)))
    cv2.moveWindow("Figure 1", 150 + int(image_keypoints.shape[1]*1.5), 150)
    cv2.imshow("Figure 1", image_keypoints)
    cv2.imwrite(r"Dataset_CvDl_Hw1_2/Q4_images/figure 1.png", image_keypoints)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def matched_keypoints():
    sift = cv2.xfeatures2d.SIFT_create()

    image_1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_1_gray, None)

    image_2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_2_gray, None)

    
    #Brute-Force Matching with SIFT Descriptors and Ratio Test
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, 2)

    # Apply ratio test explained by D.Lowe
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    image_match = cv2.drawMatchesKnn(image_1_gray, keypoints_1, image_2_gray, keypoints_2, good, None, (0, 255, 255), (0, 255, 0), None, cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    '''
    #FLANN based Matcher
    # FLANN parameters(SIFT, SURF)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #the number of times the trees in the index should be recursively traversed
    search_params = dict(checks = 50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, 2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))] #all mask
    # ratio test as per Lowe's paper
    #emumerate: (index, matches)
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0, 255, 255),
                   singlePointColor = (0, 255, 0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

    #* positional arguments
    #** keyword arguments(dict)
    image_match = cv2.drawMatchesKnn(image_1_gray, keypoints_1, image_2_gray, keypoints_2, matches, None, **draw_params)
    '''

    cv2.namedWindow("traffics.png", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("traffics.png", (int(image_1.shape[1]*1.5), int(image_1.shape[0]*1.5)))
    cv2.moveWindow("traffics.png", 100, 100)
    cv2.imshow("traffics.png", image_1)

    cv2.namedWindow("ambulance.png", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ambulance.png", (int(image_2.shape[1]*1.5), int(image_2.shape[0]*1.5)))
    cv2.moveWindow("ambulance.png", 100 + int(image_1.shape[1]*1.5), 100)
    cv2.imshow("ambulance.png", image_2)

    cv2.namedWindow("Figure 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Figure 2", (int(image_match.shape[1]*1.5), int(image_match.shape[0]*1.5)))
    cv2.moveWindow("Figure 2", 100 + int(image_1.shape[1]*1.5) + int(image_2.shape[1]*1.5), 100)
    cv2.imshow("Figure 2", image_match)
    cv2.imwrite(r"Dataset_CvDl_Hw1_2/Q4_images/figure 2.png", image_match)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("2022 CvDl Hw1")
        self.resize(270, 360)

        self.Widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.Widget)
        
        self.GroupBoxLayOut = QtWidgets.QHBoxLayout()
        self.GroupBoxLayOut.setContentsMargins(30, 30, 30, 30)

        self.GroupBox = QtWidgets.QGroupBox("4.SIFT", self)

        self.ButtonLayOut = QtWidgets.QVBoxLayout()
        self.ButtonLayOut.setContentsMargins(20, 0, 20, 0)
        
        self.PushButton1 = QtWidgets.QPushButton("Load Image 1", self)
        self.PushButton1.setStyleSheet("padding:10px")
        
        self.PushButton2 = QtWidgets.QPushButton("Load Image 2", self)
        self.PushButton2.setStyleSheet("padding:10px")

        self.PushButton3 = QtWidgets.QPushButton("4.1 Keypoints", self)
        self.PushButton3.setStyleSheet("padding:10px")

        self.PushButton4 = QtWidgets.QPushButton("4.2 Matched Keypoints", self)
        self.PushButton4.setStyleSheet("padding:10px")
        

        self.ButtonLayOut.addWidget(self.PushButton1, 1)
        self.ButtonLayOut.addWidget(self.PushButton2, 1)
        self.ButtonLayOut.addWidget(self.PushButton3, 1)
        self.ButtonLayOut.addWidget(self.PushButton4, 1)
        self.GroupBox.setLayout(self.ButtonLayOut)

        self.GroupBoxLayOut.addWidget(self.GroupBox)        
        self.Widget.setLayout(self.GroupBoxLayOut)


        self.PushButton1.clicked.connect(load_image_1)
        self.PushButton2.clicked.connect(load_image_2)
        self.PushButton3.clicked.connect(keypoints)
        self.PushButton4.clicked.connect(matched_keypoints)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
