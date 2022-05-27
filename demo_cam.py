"""
Author: Loc Nguyen
 
"""

import cv2 
from PIL import Image
import os
from tqdm import tqdm

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation,estimateAllTranslation2
from applyGeometricTransformation import applyGeometricTransformation
import time
import numpy as np

rects1 = list()
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
cam = cv2.VideoCapture(6)#realsense

#feature map
    
while True:
    ret_val, img = cam.read() 
    if len(rects1)==0:
        im_draw=img.copy()
        print("Press 'b' to select boxes.")      
        cv2.putText(im_draw,"Press 'b' to select boxes.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=0.5,color=(0,255,255),thickness=2)  
        cv2.imshow('Object Tracking', im_draw)

    k=cv2.waitKey(1)
    if k == 27: 
        break  # esc to quit
    if k== ord('b'):    
        im_draw=img.copy()
        cv2.imshow('Object Tracking', im_draw) 
        rects = select_exemplar_rois(im_draw,'Object Tracking') 
         
        rects1 = list()
        for rect in rects:
            y1, x1, y2, x2 = rect
            rects1.append([y1, x1, y2, x2]) 
            
   
        print("Bounding boxes: ", end="")
        print(rects1) 
        pre_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  
        n_object=len(rects1)
        bboxs = np.empty((n_object,4,2), dtype=float)
        for i in range(n_object):
            y1, x1, y2, x2 = rects1[i]
            xmin, ymin, boxw, boxh = x1,y1,x2-x1,y2-y1
            bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
        
        pre_bboxs=bboxs
        startXs,startYs = getFeatures(pre_img,bboxs,use_shi=False)
    

    #ONLINE INFERENCE
    if len(rects1)>0:
        cur_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  
        start_t=time.time()
        newXs, newYs = estimateAllTranslation2(startXs, startYs, pre_img, cur_img)
        Xs, Ys ,cur_bboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, pre_bboxs)
        print('Processing time =',time.time()-start_t)
        # update coordinates
        startXs = Xs
        startYs = Ys
        pre_img=cur_img.copy()
         
        # update feature points as required
        n_features_left = np.sum(Xs!=-1)
        print('# of Features: %d'%n_features_left)
        if n_features_left < 15:
            print('Generate New Features')
            startXs,startYs = getFeatures(cur_img,bboxs[i])

        pre_bboxs=cur_bboxs

        # draw bounding box and visualize feature point for each object
        frames_draw  = img.copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))
            frames_draw  = cv2.rectangle(frames_draw , (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,0), 2)
            for k in range(startXs.shape[0]):
                frames_draw = cv2.circle(frames_draw , (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)
        
        # imshow if to play the result in real time
       
        cv2.imshow("win",frames_draw[i])
        cv2.waitKey(10)
        if save_to_file:
            out.write(frames_draw[i])
       cv2.imshow('Object Tracking', img)   

cv2.destroyAllWindows()
 
 




