#Author: Loc Nguyen
#Tracking bounding boxes by KLT 
import numpy as np
import cv2  
from getFeatures import getFeatures
from applyGeometricTransformation import applyGeometricTransformation
import time 
import random
pre_img=None
cur_img=None
def select_exemplar_rois(image,window_name):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI(window_name, image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print("Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois  
    
#WEBCAM  
cap = cv2.VideoCapture(6)#realsense
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
rects=list()

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    if len(rects)==0: 
        im_draw=frame.copy() 
        cv2.putText(im_draw,"Press 'b' to select boxes.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.imshow('Object Tracking', im_draw) 
    
    k=cv2.waitKey(1)
    if k == 27: 
        break  # esc to quit
    if k== ord('c'):  
        mask = np.zeros_like(frame)

    #Select bounding boxes
    if k== ord('b'):    
        im_draw=frame.copy() 
        cv2.putText(im_draw,"Press 'n' to select boxes.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.putText(im_draw,"then press 'space' to confirm",(10,80),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.imshow('Object Tracking', im_draw) 
        rects = select_exemplar_rois(im_draw,'Object Tracking') 
          
        # Take first frame and find corners in it 
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        n_object=len(rects)
        bboxs  = np.empty((n_object,4,2), dtype=float)
        for i,rect in enumerate(rects):
            y1, x1, y2, x2 = rect  
            xmin, ymin, boxw, boxh=x1,y1,x2-x1,y2-y1
            bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    
        print("Bounding boxes: ", end="")  
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params) 
        startXs,startYs = getFeatures(old_gray,bboxs,use_shi=False)
        
        p0  = np.zeros(shape=(int(startXs.shape[0]*startXs.shape[1] ),1,2), dtype=np.float32) #np.array((startXs.shape[0],1,2), dtype=float)
        for i in range(startXs.shape[0]):#number of keypoints
            for j in range(startXs.shape[1]):#number of bboxes
                p0[j*int(startXs.shape[0])+i][0][0]=startXs[i,j]
                p0[j*int(startXs.shape[0])+i][0][1]=startYs[i,j]
        for i in range(p0.shape[0]):
            cv2.circle(im_draw,(int(p0[i][0][0]),int(p0[i][0][1])),radius=3,color=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),thickness=-1)
        cv2.imshow('Object Tracking', im_draw) 
        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)

        #init for next frame
        Xs=np.copy(startXs)
        Ys=np.copy(startYs)
        newXs=np.copy(startXs)
        newYs=np.copy(startYs)
        newbboxs=np.copy(bboxs)

    #Online tracking
    if len(rects)>0:
        start_t=time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # calculate optical flow 
        p0  = np.zeros(shape=(int(startXs.shape[0]*startXs.shape[1] ),1,2), dtype=np.float32) 
        for i in range(startXs.shape[0]):#number of keypoints
            for j in range(startXs.shape[1]):#number of bboxes
                p0[j*int(startXs.shape[0])+i][0][0]=startXs[i,j]
                p0[j*int(startXs.shape[0])+i][0][1]=startYs[i,j]
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # # Select good points
        # if p1 is not None:
        #     good_new = p1[st==1]
        #     good_old = p0[st==1]

        # # draw the tracks
        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #     frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        # img = cv2.add(frame, mask) 
        # cv2.putText(img,"Press 'c' to reset drawing.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        
        # cv2.imshow('frame', img) 
  
        #=================================
        #copy back to array
        for i in range(startXs.shape[0]):#number of keypoints
            for j in range(startXs.shape[1]):#number of bboxes
                newXs[i,j]=p1[j*int(startXs.shape[0])+i][0][0]
                newYs[i,j]=p1[j*int(startXs.shape[0])+i][0][1]
        
        Xs, Ys ,newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
        
        #in case of failed transform
        array_sum = np.sum(newbboxs)
        if np.isnan(array_sum):
            newbboxs=np.copy(bboxs)

        bboxs=np.copy(newbboxs)
        #bboxs=np.copy(newbboxs)
        # update coordinates
        startXs = np.copy(Xs)
        startYs = np.copy(Ys)        

        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(frame_gray,newbboxs)
            p0  = np.zeros(shape=(int(startXs.shape[0]*startXs.shape[1] ),1,2), dtype=np.float32) #np.array((startXs.shape[0],1,2), dtype=float)
            for i in range(startXs.shape[0]):#number of keypoints
                for j in range(startXs.shape[1]):#number of bboxes
                    p0[j*int(startXs.shape[0])+i][0][0]=startXs[i,j]
                    p0[j*int(startXs.shape[0])+i][0][1]=startYs[i,j]
        #draw bounding box and visualize feature point for each object
        frames_draw  = frame.copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbboxs[j,:,:].astype(int))
            frames_draw = cv2.rectangle(frames_draw , (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            for k in range(startXs.shape[0]):
                frames_draw  = cv2.circle(frames_draw, (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)
        cv2.imshow('frame2', frames_draw) 

        #=================================
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0=p1
        # p0 = good_new.reshape(-1, 1, 2)  
        # for i in range(startXs.shape[0]):#number of keypoints
        #     for j in range(startXs.shape[1]):#number of bboxes
        #         startXs[i,j]=p0[j*int(startXs.shape[0])+i][0][0]
        #         startYs[i,j]=p0[j*int(startXs.shape[0])+i][0][1]

        print("all time spent=",time.time()-start_t)
    
cv2.destroyAllWindows()