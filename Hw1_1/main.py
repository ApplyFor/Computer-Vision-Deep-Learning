import numpy as np
import cv2

import matplotlib

from UI import Ui_widget #UI.py
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

import os

Q1_images = []
Q2_images = []
def load_images():
    #可以直接存取無法直接修改
    global Q1_images
    global Q2_images

    if Q1_images:
        Q1_images = []
    folder = 'Dataset_CvDl_Hw1/Q1_Image/'
    for filename in os.listdir(folder): #取得指定目錄中所有的檔案與子目錄名稱
        img = cv2.imread(os.path.join(folder, filename)) #os.path.join folder\filename
        if img is not None:
            Q1_images.append(img)

    if Q2_images:
        Q2_images = []
    folder = 'Dataset_CvDl_Hw1/Q2_Image/'
    for filename in os.listdir(folder): #取得指定目錄中所有的檔案與子目錄名稱
        img = cv2.imread(os.path.join(folder, filename)) #os.path.join folder\filename
        if img is not None:
            Q2_images.append(img)

imL = []
def load_image_L():
    global imL
    #raw string
    imL = cv2.imread(r'Dataset_CvDl_Hw1/Q3_Image/imL.png')

imR = []
def load_image_R():
    global imR
    #raw string
    imR = cv2.imread(r'Dataset_CvDl_Hw1/Q3_Image/imR.png')

def find_corners(): #11,8(內部)
    w = 11
    h = 8
    #TERM_CRITERIA_EPS 誤差满足epsilon停止
    #TERM_CRITERIA_MAX_ITER 迭代次數超過max_iter停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) #迭代停止(type,max_iter,epsilon)

    for img in Q1_images:
        #ret retval return value
        ret, corners = cv2.findChessboardCorners(img, (w,h), None) #Find chessboard corners
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) #Get Subpixel accuracy on those corners
            tmp = img.copy()
            cv2.drawChessboardCorners(tmp, (w,h), accurate_corners, ret) #Draw it

            cv2.namedWindow('find_corners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('find_corners', (int(tmp.shape[1]/3), int(tmp.shape[0]/3)))
            cv2.imshow('find_corners', tmp)
            cv2.waitKey(500) #show each picture 0.5 seconds
            cv2.destroyAllWindows()

def find_intrinsic():
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    #print(np.mgrid[0:w, 0:h])
    #print(np.mgrid[0:w, 0:h].shape)
    #print(np.mgrid[0:w, 0:h].T)
    #print(np.mgrid[0:w, 0:h].T.shape)
    #print(np.mgrid[0:w, 0:h].T.reshape(-1, 2)) #-1:自動計算 =(w*h,2)
    #print(np.mgrid[0:w, 0:h].T.reshape(-1, 2).shape)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)   #x,y,0
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane

    for img in Q1_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None) #Find chessboard corners
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)
    
    #ret 誤差
    #mtx 內部參數
    #dist 畸變參數
    #rvecs 世界坐標系到相機坐標系的旋轉参數(rotation vectors)
    #tvecs 平移參數(translation vectors)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Intrinsic:")
    print(mtx)

def find_extrinsic(number):
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = [] 
    imgpoints = []

    for img in Q1_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    R, jacobian = cv2.Rodrigues(rvecs[number]) #Rodrigues Transform
    E = np.concatenate((R,tvecs[number]), axis=1) #[R|T]
    print("Extrinsic:")
    print(E)

def find_distortion():
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for img in Q1_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Distortion:")
    print(dist)

def show_result():
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    for img in Q1_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    for img in Q1_images:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx) # undistort
        # crop the image(alpha = 1)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv2.namedWindow('Distorted image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Distorted image', (int(img.shape[1]/3), int(img.shape[0]/3)))
        cv2.moveWindow('Distorted image', 100, 100)
        cv2.imshow('Distorted image', img)
        
        cv2.namedWindow('Undistorted image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Undistorted image', (int(dst.shape[1]/3), int(dst.shape[0]/3)))
        cv2.moveWindow('Undistorted image', 100 + int(img.shape[1]/3), 100)
        cv2.imshow('Undistorted image', dst)
        
        cv2.waitKey(500) #show each picture 0.5 seconds
        cv2.destroyAllWindows()

def augmented_reality(image, projectpoints):
    #([point_index][0][0:x 1:y])
    #print(projectpoints[0][0][0], projectpoints[0][0][1], projectpoints[1][0][0], projectpoints[1][0][1])
    cv2.line(image, 
            (int(projectpoints[0][0][0]),int(projectpoints[0][0][1])), 
            (int(projectpoints[1][0][0]),int(projectpoints[1][0][1])), 
            (0, 0, 255), 10)

def show_words_on_board(word):
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    objectpoints = []
    frame = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]], np.float32)
    fs = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
    i = 0 #index for frame
    for char in word:
        ch = fs.getNode(char.upper()).mat()
        #print(ch) (line_number, 2, 3)
        objectpoints.append(ch+frame[i])
        #print(objectpoints)
        i += 1

    for img in Q2_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    i = 0 #index for rotation vectors and translation vectors
    for img in Q2_images:
        tmp = img.copy()
        for word_frame in objectpoints:
            for line in word_frame: #(2, 3)
                imagepoints, jacobian = cv2.projectPoints(line, rvecs[i], tvecs[i], mtx, dist)
                #print(line)
                #print(imagepoints) #(2, 1, 2)
                augmented_reality(tmp, imagepoints)
        
        cv2.namedWindow('AR_onboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AR_onboard', (int(tmp.shape[1]/3), int(tmp.shape[0]/3)))
        cv2.imshow('AR_onboard', tmp)
        
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        i += 1

def show_words_vertically(word):
    w = 11
    h = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    objp = np.zeros((w*h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []

    objectpoints = []
    frame = np.array([[7, 5, 0], [4, 5, 0], [1, 5, 0], [7, 2, 0], [4, 2, 0], [1, 2, 0]], np.float32)
    fs = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
    i = 0 #index for frame
    for char in word:
        ch = fs.getNode(char.upper()).mat()
        #print(ch) (line_number, 2, 3)
        objectpoints.append(ch+frame[i])
        #print(objectpoints)
        i += 1

    for img in Q2_images:
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            accurate_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(accurate_corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    i = 0 #index for rotation vectors and translation vectors
    for img in Q2_images:
        tmp = img.copy()
        for word_frame in objectpoints:
            for line in word_frame: #(2, 3)
                imagepoints, jacobian = cv2.projectPoints(line, rvecs[i], tvecs[i], mtx, dist)
                #print(line)
                #print(imagepoints) #(2, 1, 2)
                augmented_reality(tmp, imagepoints)
        
        cv2.namedWindow('AR_vertical', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AR_vertical', (int(tmp.shape[1]/3), int(tmp.shape[0]/3)))
        cv2.imshow('AR_vertical', tmp)
        
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        i += 1

def stereo_disparity_map():
    grayL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY);

    #numDisparities
    #blockSize
    stereo = cv2.StereoBM_create(256, 25)
    disparity = stereo.compute(grayL, grayR) #16-bit fixed-point(where each disparity value has 4 fractional bits)
    result = disparity_map(imL, imR, disparity)
    result.show()

class disparity_map():
    def __init__(self, imL, imR, disparity):
        self.imgL = imL
        self.imgR = imR
        self.tmp = imR.copy()
        self.baseline = 342.789 #mm
        self.focal_length = 4019.284 #pixel
        self.cright_cleft = 279.184 #pixel
        self.disp = disparity #pixel
        self.vdisp = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) #Map disparity range to gray value range 0~255 for the purpose of visualization
    
    def depth(self, disparity):
        return self.focal_length*self.baseline/(disparity)

    def correspondence(self, event, x, y, flags, param):   
        if event == cv2.EVENT_LBUTTONDOWN:
            #depth = self.depth(self.disp[y][x])
            self.imgR = self.tmp.copy()
            if(self.vdisp[y][x] == 0):
                return

            #y rectified
            #print("disparity: ",self.disp[y][x])
            #print("normalized disparity: ",self.vdisp[y][x], end = "\n\n")
            xR = x - self.vdisp[y][x];
            cv2.circle(self.imgR, (xR, y), 10, (0, 255, 0), -1)
            
    def show(self):
        cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imgL', (int(self.imgL.shape[1]/4.5), int(self.imgL.shape[0]/4.5)))
        cv2.moveWindow('imgL', 45, 300)

        cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imgR_dot', (int(self.imgR.shape[1]/4.5), int(self.imgR.shape[0]/4.5)))
        cv2.moveWindow('imgR_dot', 45 + int(self.imgL.shape[1]/4.5), 300)

        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disparity', (int(self.vdisp.shape[1]/4.5), int(self.vdisp.shape[0]/4.5)))
        cv2.moveWindow('disparity', 45 + int(self.imgL.shape[1]/4.5) + int(self.imgR.shape[1]/4.5), 300)
        
        while(1):
            cv2.imshow('imgL', self.imgL)
            
            cv2.imshow('imgR_dot', self.imgR)

            cv2.imshow('disparity', self.vdisp)

            cv2.setMouseCallback('imgL', self.correspondence)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_widget()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(load_images)
        self.ui.pushButton_2.clicked.connect(load_image_L)
        self.ui.pushButton_3.clicked.connect(load_image_R)

        self.ui.pushButton_4.clicked.connect(find_corners)
        self.ui.pushButton_5.clicked.connect(find_intrinsic)
        self.ui.pushButton_7.clicked.connect(self.extrinsic)
        self.ui.pushButton_8.clicked.connect(find_distortion)
        self.ui.pushButton_9.clicked.connect(show_result)

        self.ui.pushButton_10.clicked.connect(self.board)
        self.ui.pushButton_11.clicked.connect(self.vertical)

        self.ui.pushButton_12.clicked.connect(stereo_disparity_map)

    def extrinsic(self):
        num = self.ui.comboBox.currentIndex()
        find_extrinsic(num)

    def board(self):
        w = self.ui.textEdit.toPlainText()
        show_words_on_board(w)
    def vertical(self):
        w = self.ui.textEdit.toPlainText()
        show_words_vertically(w)

if __name__ == "__main__": #主執行檔
    app = QtWidgets.QApplication(sys.argv)
    widget = GUI()
    widget.show() #顯示視窗
    sys.exit(app.exec_())