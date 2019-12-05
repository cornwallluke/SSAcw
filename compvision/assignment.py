import yolo.yolomodule as yl
import stereodisparity.disparitymodule as dp
import numpy as np
import cv2
import csv
import os
from time import time as ti
cv2.namedWindow('yeet')
cv2.namedWindow('yeet2')

#fails here 
#TTBB-durham-02-10-17-sub10/left-images/1506943261.483305_L.png


cv2.namedWindow('yeet3')
yeet = 'yeet'
master_path_to_dataset = "TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left))

def gothroughfiles():
    for filename_left in left_file_list:
        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        # for sanity print out these filenames

        print(full_path_filename_left)
        # print(full_path_filename_right)
        # print()
        yield cv2.imread(full_path_filename_left),cv2.imread(full_path_filename_right)
imageiterator = gothroughfiles()


clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(32,32))
mask = cv2.imread('yolo/mask.png')
maskbw = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
left, right = next (imageiterator)
while left is not None:
    # leftfname = 'TTBB-durham-02-10-17-sub10/left-images/1506942689.476653_L.png'
    # leftfname = 'TTBB-durham-02-10-17-sub10/left-images/1506943442.478606_L.png'
    # leftoriginal = cv2.imread(leftfname)
    # left = leftoriginal
    # right = cv2.imread(leftfname.replace("_L", "_R").replace('left','right'))
    
    leftoriginal = left
    t = ti()
    classIDs,confidences,boxes = yl.objectsInFrame(leftoriginal,0.8)
    print("yolo time: "+str(ti()-t))
    t = ti()
    # left = cv2.bitwise_and(leftoriginal,mask)
    # right = cv2.bitwise_and(right,mask)
    leftG = cv2.fastNlMeansDenoising(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),h=5)
    rightG = cv2.fastNlMeansDenoising(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),h=5)










    
    left = clahe.apply(leftG)#cv2.equalizeHist(leftG)
    right = clahe.apply(rightG)#cv2.equalizeHist(rightG)
    # left = np.power(left, 0.75).astype('uint8')
    # right = np.power(right, 0.75).astype('uint8')


    print("preproc time: "+str(ti()-t))
    t=ti()
    
    
    disp, dispscaled,rawdisp = dp.getDisparity(left,right)
    _, fdispscaled, _ = dp.getDisparity(np.flip(right,1),np.flip(left,1))
    fdispscaled = np.flip(fdispscaled,1)

    pointsmap = np.nan_to_num((399.9745178222656*0.2090607502/dispscaled),nan=0,posinf = 0,neginf=0).astype(np.uint8)
    fpoints = np.nan_to_num((399.9745178222656*0.2090607502/fdispscaled),nan=0,posinf = 0,neginf=0).astype(np.uint8)
    #points,pointsmap = dp.project_disparity_to_3d(dispscaled, leftoriginal)
    combined = np.zeros(pointsmap.shape).astype(np.uint8)
    combined[:,:128] = fpoints[:,:128]
    combined[:,-128:] = pointsmap[:,-128:]
    combined[:,128:-128] = fpoints[:,128:-128]/2+pointsmap[:,128:-128]/2

    # pointsmap = cv2.bitwise_and(clahe.apply(pointsmap),maskbw)
    pointsmap = cv2.bitwise_and(cv2.equalizeHist(pointsmap),maskbw)
    fpoints = cv2.bitwise_and(cv2.equalizeHist(fpoints),maskbw)
    combined = cv2.bitwise_and(cv2.equalizeHist(combined),maskbw)
    
    #altpointsmap = dp.threed2twod(points,left.shape[:2])

    print("dispr time: "+str(ti()-t))
    for o in range(len(boxes)):
        yl.drawPred(combined,yl.getName(classIDs[o]),confidences[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))


    cv2.imshow('yeet',leftG)    
    #cv2.imshow('yeet2',cv2.addWeighted(cv2.cvtColor(pointsmap,cv2.COLOR_GRAY2BGR),0.5,leftoriginal,0.5,1))
    cv2.imshow('yeet2',fpoints)
    cv2.imshow('yeet3',combined)
    cv2.waitKey(10)

    try:
        left,right = next(imageiterator)
    except:
        left = None
cv2.destroyAllWindows()
