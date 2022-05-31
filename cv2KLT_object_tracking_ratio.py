#Item: Implement simple tracking bounding boxes by opencv KLT
#Author: Loc Nguyen
# 2022
#Steps:
#1. Select ROIs
#2. Run tracking for multiple ROIs

import numpy as np
import cv2   
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max
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
def getFeatures(img,bbox,use_shi=True): 
    n_object = np.shape(bbox)[0]
    N = 0
    temp = np.empty((n_object,),dtype=np.ndarray)   # temporary storage of x,y coordinates
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)
        coordinates = peak_local_max(corner_response,num_peaks=20,exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    x = np.full((N,n_object),-1)
    y = np.full((N,n_object),-1)
    for i in range(n_object):
        n_feature = temp[i].shape[0]
        x[0:n_feature,i] = temp[i][:,1]
        y[0:n_feature,i] = temp[i][:,0]
    return x,y    
#WEBCAM  
cap = cv2.VideoCapture(4)#realsense
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
ratios=list()#list of sx,sy for keypoints and BBoxes
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
    if k== ord('b'):    
        im_draw=frame.copy() 
        cv2.putText(im_draw,"Press 'n' to select boxes.",(10,50),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.putText(im_draw,"then press 'space' to confirm",(10,80),cv2.FONT_HERSHEY_SIMPLEX ,fontScale=1,color=(0,255,255),thickness=2)  
        cv2.imshow('Object Tracking', im_draw) 
        rects = select_exemplar_rois(im_draw,'Object Tracking') 
        
        start_t=time.time()
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
    
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params) 
        startXs,startYs = getFeatures(old_gray,bboxs,use_shi=False) 

        p0  = np.zeros(shape=(int(startXs.shape[0]*startXs.shape[1] ),1,2), dtype=np.float32) #np.array((startXs.shape[0],1,2), dtype=float)
        for i in range(startXs.shape[0]):#number of keypoints
            for j in range(startXs.shape[1]):#number of bboxes
                p0[j*int(startXs.shape[0])+i][0][0]=startXs[i,j]
                p0[j*int(startXs.shape[0])+i][0][1]=startYs[i,j] 
        
        #mask whole array
        p0_mask  = np.ones(int(startXs.shape[0]*startXs.shape[1]), dtype=np.int32)
        p0_index=np.array(np.where(p0_mask>0)).astype(np.int32).reshape((-1)) 
        p0_valid=p0 #init


        #get offset ratio
        for j,rect in enumerate(rects):
            y1, x1, y2, x2 = rect 
            minx=np.min(startXs[:,j])
            miny=np.min(startYs[:,j])
            maxx=np.max(startXs[:,j])
            maxy=np.max(startYs[:,j])

            r_top=(miny-y1)/(maxy-miny)
            r_bottom=(y2-maxy)/(maxy-miny)
            r_left=(minx-x1)/(maxx-minx)
            r_right=(x2-maxx)/(maxx-minx)
            ratios.append([r_top,r_bottom,r_left,r_right])

        print("get features time=",time.time()-start_t)
        for i in range(p0.shape[0]):
            cv2.circle(im_draw,(int(p0[i][0][0]),int(p0[i][0][1])),radius=3,color=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),thickness=-1)
        cv2.imshow('Object Tracking', im_draw) 
         
    if len(rects)>0:
        start_t=time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         
        # calculate optical flow 
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_valid, None, **lk_params)
          
        # Select good points
        if p1 is not None:
            good_new_all = p1[st==1]
            good_old_all = p0_valid[st==1] 

            #set indices of new invalids
            p0_mask[p0_index[st.reshape((-1))==0]]=0
            p0_index=np.array(np.where(p0_mask>0)).astype(np.int32).reshape((-1))  #mapping index for good_new_all

            #transform bboxes
            for j in range(len(rects)): 
                range_min=j*int(startXs.shape[0])
                range_max=(j+1)*int(startXs.shape[0])-1
                selected_indices=np.where((p0_index >= range_min) & (p0_index <range_max))

                if not selected_indices or len(selected_indices[0])<3:
                    print(j,": <3 not enough points")
                    continue         

                good_old=good_old_all[selected_indices]
                good_new=good_new_all[selected_indices]
                  
                newx_min=np.min(good_new[:,0])
                newx_max=np.max(good_new[:,0])
                newy_min=np.min(good_new[:,1])
                newy_max=np.max(good_new[:,1]) 
                  
                #display tracking points
                for i, new in enumerate(good_new):
                    a, b = new.ravel() 
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1) 
                
                #apply ratio
                y1, x1, y2, x2 = rects[j]      
                r_top,r_bottom,r_left,r_right = ratios[j]      

                newy1=int(newy_min-r_top*(newy_max-newy_min))
                newy2=int(newy_max+r_bottom*(newy_max-newy_min))
                newx1=int(newx_min-r_left*(newx_max-newx_min))
                newx2=int(newx_max+r_right*(newx_max-newx_min)) 
                rects[j]=[newy1,newx1,newy2,newx2] 
                frame = cv2.rectangle(frame, (newx1, newy1), (newx2, newy2), color[j].tolist(), thickness=2) 
            
            print("tracking time =",f'{time.time()-start_t:n}')    
            cv2.imshow('frame', frame) 

            # Now update the previous frame and previous points 
            p0_valid = good_new_all.reshape(-1, 1, 2) 
            old_gray = frame_gray.copy() 
        else:
            print("tracking failed!") 
    
cv2.destroyAllWindows()