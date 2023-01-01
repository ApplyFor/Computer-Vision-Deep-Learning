import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def background_subtraction(video):
    #backSub = cv2.bgsegm.createBackgroundSubtractorMOG(25)
    gaussian = []
    foreground = []

    if not video:
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Cannot open ' + video[video.rfind(r'/')+1:])
        return

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        '''
        fgmask = backSub.apply(frame)
        fg = cv2.bitwise_and(frame, frame, None, fgmask)
        fgmask = np.dstack((fgmask, fgmask, fgmask))
        out = cv2.hconcat([frame, fgmask, fg])
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(gray)

        if(len(gaussian) < 25):
            gaussian.append(gray)
            if(len(gaussian) == 25):
                gaussian = np.array(gaussian)
                mean = np.mean(gaussian, axis= 0)
                std = np.std(gaussian, axis=0)
                std[std < 5] = 5
        else:
            mask[np.abs(gray - mean) > 5 * std] = 255
        
        foreground = cv2.bitwise_and(frame, frame, None, mask)

        mask = np.expand_dims(mask, axis=2)
        mask = np.concatenate((mask, mask, mask) ,axis=2)
        
        out = cv2.hconcat([frame, mask, foreground])
        
        cv2.imshow("frame", out)
        if cv2.waitKey(30) & 0xff == ord('q'): # &0xff防止BUG
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocessing(video):
    if not video:
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Cannot open ' + video[video.rfind(r'/')+1:])
        return

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive first frame. Exiting ...")
            return

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 35
        params.maxArea = 90
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 1.0
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        params.maxInertiaRatio = 1.0

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(frame)

        image = frame.copy()
        points = cv2.KeyPoint_convert(keypoints)
        for point in points:
            image = cv2.rectangle(image, tuple(point-6), tuple(point+6), (0, 0, 255))
            image = cv2.line(image, (int(point[0]-6), int(point[1])), (int(point[0]+6), int(point[1])), (0, 0, 255))
            image = cv2.line(image, (int(point[0]), int(point[1]-6)), (int(point[0]), int(point[1]+6)), (0, 0, 255))

        cv2.imshow("Circle detect", image)
        if cv2.waitKey(0) & 0xff == 27: #ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

    return points

def video_tracking(video, points):
    if not video:
        return

    if type(points) == list:
        return
    if not points.any():
        return

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print('Cannot open ' + video[video.rfind(r'/')+1:])
        return

    # Take first frame and find corners in it
    ret, frame = cap.read()
    prevGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prevGray)
    mask = np.dstack((mask, mask, mask))
    
    p0 = np.expand_dims(points,axis=1) #(7,2)->(7,1,2)

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        # if frame is read correctly ret is True
        if not ret:
            print('No frames grabbed!')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(prevGray, gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[status==1]
            good_old = p0[status==1]

        # draw the tracks
        for new, old in zip(good_new, good_old): #zip(new, old)
            a,b = new.ravel() #1D-array
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 255), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 255), -1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame', img)
        if cv2.waitKey(30) & 0xff == 27: #ESC
            break


        # Now update the previous frame and previous points
        prevGray = gray
        p0 = good_new.reshape(-1, 1, 2) #(n,1,2)

    cap.release()
    cv2.destroyAllWindows()

def perspective_transform(imagename, videoname):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()

    if not imagename:
        return
    #不能讀取中文路徑
    image = cv2.imread(imagename)

    if not videoname:
        return

    cap = cv2.VideoCapture(videoname)
    if not cap.isOpened():
        print('Cannot open ' + videoname[videoname.rfind(r'/')+1:])
        return

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        # if frame is read correctly ret is True
        if not ret:
            print('No frames grabbed!')
            return

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters = parameters)
        #cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        Id1_sequence = np.where(markerIds == 1)[0]
        Id2_sequence = np.where(markerIds == 2)[0]
        Id3_sequence = np.where(markerIds == 3)[0]
        Id4_sequence = np.where(markerIds == 4)[0]

        if not (len(Id1_sequence) and len(Id2_sequence) and len(Id3_sequence) and len(Id4_sequence)):
            #print(Id1_sequence, Id2_sequence, Id3_sequence, Id4_sequence, end = ' ')
            #print()
            continue

        src_pts = []
        dst_pts = []

        #the top left corner
        src_pts.append([0, 0])
        dst_pts.append((markerCorners[Id1_sequence[0]])[0][0])
        #the top right corner
        src_pts.append([image.shape[1], 0])
        dst_pts.append((markerCorners[Id2_sequence[0]])[0][1])
        #the bottom right corner
        src_pts.append([image.shape[1], image.shape[0]])
        dst_pts.append((markerCorners[Id3_sequence[0]])[0][2])
        #the bottom left corner
        src_pts.append([0, image.shape[0]])
        dst_pts.append((markerCorners[Id4_sequence[0]])[0][3])

        M, mask = cv2.findHomography(np.asfarray(src_pts), np.asfarray(dst_pts), cv2.RANSAC, 5.0)
        dst = cv2.warpPerspective(image, M, (frame.shape[1], frame.shape[0]))
        
        logo_mask = np.zeros_like(frame)
        logo_mask[dst[:,:,:] > 0] = 255
        logo_mask = cv2.bitwise_not(logo_mask) #reverse
        #cv2.imshow("mask", logo_mask)
        img = cv2.bitwise_and(frame, logo_mask)
        #cv2.imshow("frame", img)
        img = cv2.bitwise_or(img, dst)

        out = cv2.hconcat([frame, img])

        cv2.namedWindow("out", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("out", (int(out.shape[1]/2.5), int(out.shape[0]/2.5)))
        cv2.imshow("out", out)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def image_reconstruction(image, width, height):
    if not image:
        return

    flat = []
    for img in image:
        flat.append(img.flatten())
    #print('flatten image', flat[0].shape)

    n = len(image)
    #print('n =', n)
    pca = PCA(n_components = int(0.9*n))
    reduce_ = pca.fit_transform(flat)
    reconstruct = pca.inverse_transform(reduce_)
    reconstruct = cv2.normalize(reconstruct, None, 0, 255, cv2.NORM_MINMAX)
    reconstruct = np.reshape(reconstruct.astype(np.uint8), (n, width, height, 3))
    #print('reconstructed image', reconstruct.shape)

    half = n//2 + n % 2
    plt.figure()
    for i in range(n):
        if i < half:
            plt.subplot(4, half, 1+i)
            plt.imshow(image[i])
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('origin')
            plt.subplot(4, half, 1+half+i)
            plt.imshow(reconstruct[i])
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                plt.ylabel('reconstruction')
        else:
            plt.subplot(4, half, 1+half+i)
            plt.imshow(image[i])
            plt.xticks([])
            plt.yticks([])
            if i == half:
                plt.ylabel('origin')
            plt.subplot(4, half, 1+2*half+i)
            plt.imshow(reconstruct[i])
            plt.xticks([])
            plt.yticks([])
            if i == half:
                plt.ylabel('reconstruction')
    plt.show()

    return reconstruct

def reconstruction_error_computation(image, reconstruct):
    if type(reconstruct) == list:
        return

    RE = []
    for i in range(len(image)):
        #numpy overflow
        origin_gray = cv2.cvtColor(image[i], cv2.COLOR_RGB2GRAY).tolist()
        reconstruction_gray = cv2.cvtColor(reconstruct[i], cv2.COLOR_RGB2GRAY).tolist()
        #print(np.shape(origin_gray), np.shape(reconstruction_gray))

        total = 0

        
        for i in range(np.shape(reconstruction_gray)[0]):
            for j in range(np.shape(reconstruction_gray)[1]):
                error = origin_gray[i][j] - reconstruction_gray[i][j]
                error = error ** 2
                total += error
        
        total = total ** 0.5
        RE.append(total)

    print('reconstruction error:')
    #print(RE)
    print('max error:', max(RE))
    print('min error:', min(RE))
