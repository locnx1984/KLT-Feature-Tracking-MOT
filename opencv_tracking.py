import numpy as np
import cv2  
from getFeatures import getFeatures
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
  
# cap = cv2.VideoCapture("Easy.mp4")

#WEBCAM  
cap = cv2.VideoCapture(6)#realsense
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 10000,
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
    if k== ord('b'):    
        im_draw=frame.copy() 
        cv2.putText(im_draw,"Press 'n' to select boxes.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.putText(im_draw,"then press 'space' to confirm",(10,80),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.imshow('Object Tracking', im_draw) 
        rects = select_exemplar_rois(im_draw,'Object Tracking') 
          
        # Take first frame and find corners in it 
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #mask_boxes = np.zeros_like(old_gray)
        n_object=len(rects)
        bboxs  = np.empty((n_object,4,2), dtype=float)
        for i,rect in enumerate(rects):
            y1, x1, y2, x2 = rect 
            #cv2.rectangle(mask_boxes,(x1,y1),(x2,y2),255,-1)
            xmin, ymin, boxw, boxh=x1,y1,x2-x1,y2-y1
            bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
   
        # cv2.imshow('mask Object Tracking', mask) 
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

    if len(rects)>0:
        start_t=time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow 
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
 
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        
        img = cv2.add(frame, mask) 
        cv2.putText(img,"Press 'c' to reset drawing.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        
        cv2.imshow('frame', img) 
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2) 
        print("all time spent=",time.time()-start_t)
    
cv2.destroyAllWindows()