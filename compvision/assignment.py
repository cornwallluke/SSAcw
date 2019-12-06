import yolo.yolomodule as yl
#import stereodisparity.disparitymodule as dp
import numpy as np
import cv2
import csv
import os
from time import time as ti
yeet = "yeet" #debugging purposes

master_path_to_dataset = "TTBB-durham-02-10-17-sub10" # ** need to edit this **


camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5
maxdisparity = 128
denoise = 10

window_size = 7     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

classheights = [1.7,0.8,2,1,5000,3]+[2 for i in range(80)]



pred_limit = 500
pred_weight = 0.1
class tracking:
    def __init__(self):
        self.items = {}#each item is [previous positions most recent last]
    def predict_next(self):
        # print('predicting')
        # print(self.items)
        for obj in self.items:
            for i in range(len(self.items[obj])):
                
                if len(self.items[obj][i])>1:
                    # print(self.items[obj][i])
                    self.items[obj][i].append([2*self.items[obj][i][-1][k]-self.items[obj][i][-2][k] for k in range(3)])
                else:
                    self.items[obj][i].append(self.items[obj][i][-1])
        # print(self.items)
        return self.items
    def add_frame(self,itemdict):
        # print('adding')
        # print(self.items)
        # print(itemdict)
        oldpreds = dict(self.items)
        self.items = {}
        returnsdict = {}
        for i in itemdict:
            boxes = list(list(zip(*itemdict[i]))[0])
            dists = list(list(zip(*itemdict[i]))[1])
            # print(boxes)
            self.items[i] = []
            returnsdict[i] = [boxes,dists,[]]

            if i not in oldpreds:
                
                oldpreds[i] = []
           
            o = 0
            while o< len(boxes):
                # print(boxes)
                pos = [boxes[o][0]+boxes[o][2]/2,boxes[o][1]+boxes[o][3]/2,dists[o]]
                o+=1
                # print(pos)
                # print(oldpreds[i])
                if len(oldpreds[i])>0:
                    #print([pred[-1][:2] for pred in oldpreds[i]])
                    tpos= pos[:]
                    tpos[2]*=7
                    diffs = list(enumerate([euclidian(pos,pred[-1][:2]+[pred[-1][2]*7]) for pred in oldpreds[i]]))
                    # print(diffs)
                    closest = min(diffs,key= lambda x:x[1])

                    if closest[1]<pred_limit and pos[2]>0.5:# or closest[1]>:
                        pos = [pos[h]*(1-pred_weight)+oldpreds[i][closest[0]][-1][h]*pred_weight for h in range(len(pos))]
                    if pos[2]<0.1:
                        pos[2] = distanceFromHeight(classheights[i],boxes[o-1][3])#[oldpreds[i][closest[0]][-1][h] for h in range(len(pos))]

                    tmp = oldpreds[i].pop(closest[0])[:-1]+[pos]
                    returnsdict[i][2].append(pos)
                    # print(returnsdict[i][2])
                    self.items[i].append(tmp)
                else:
                    if pos[2]<0.1:
                        pos[2] = distanceFromHeight(classheights[i],boxes[o-1][3])
                    returnsdict[i][2].append(pos)
                    self.items[i].append([pos])
        # print(self.items)
        return returnsdict
            
def euclidian(a,b):
    if len(a)==len(b):
        return sum([abs(a[i]**2-b[i]**2) for i in range(len(a))])**.5
    return None

        



class disparityfinder:
    def __init__(self,maxdisp = 128):

        self.maxdisparity = maxdisp
        self.stereoProcessor = cv2.StereoSGBM_create(0, maxdisp, 11,20*window_size**2,60*window_size**2)#21)
        #     minDisparity=0,
        #     numDisparities=maxdisp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        #     blockSize=5,
        #     P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        #     P2=32 * 3 * window_size ** 2,
        #     disp12MaxDiff=1,
        #     uniquenessRatio=15,
        #     speckleWindowSize=0,
        #     speckleRange=2,
        #     preFilterCap=63,
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        # )
    def getDisparity(self,left,right,crop_disparity = False):

        originaldisparity = self.stereoProcessor.compute(left,right)

        _, disparity = cv2.threshold(originaldisparity,0, self.maxdisparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        if (crop_disparity):
            width = np.shape(disparity)[1] #np.size(disparity_scaled, 1)
            disparity_scaled = disparity_scaled[0:544,135:width]
        return (disparity_scaled * (256. / self.maxdisparity)).astype(np.uint8), disparity_scaled, originaldisparity
class wlsdisp:
    def __init__(self,maxdisp):
        
        self.maxdisparity = maxdisp
        self.left_matcher = cv2.StereoSGBM_create(0, maxdisp, 11,20*window_size**2,60*window_size**2)#21)
        #     minDisparity=0,
        #     numDisparities=maxdisp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        #     blockSize=5,
        #     P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        #     P2=32 * 3 * window_size ** 2,
        #     disp12MaxDiff=1,
        #     uniquenessRatio=15,
        #     speckleWindowSize=0,
        #     speckleRange=2,
        #     preFilterCap=63,
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        # )
        
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)
    def getDisparity(self,left,right):
        displ = self.left_matcher.compute(left, right)  # .astype(np.float32)/16
        dispr = self.right_matcher.compute(right, left)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        return 0,(self.wls_filter.filter(displ, left, None, dispr)/16).astype(np.uint8),0
   
def gothroughfiles():
    skipto = "1506942553.483915"#"1506942485.480259"
    for filename_left in left_file_list:
        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)
        if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :
            if ((len(skipto) > 0) and not(skipto in filename_left)):
                continue
            elif ((len(skipto) > 0) and (skipto in filename_left)):
                skipto = ""
            # for sanity print out these filenames

            print(full_path_filename_left)
            # print(full_path_filename_right)
            # print()
            yield cv2.imread(full_path_filename_left),cv2.imread(full_path_filename_right)
def distanceFromHeight(estHeight, heightPx):
    # print(heightPx)
    # print(estHeight)
    # print(camera_focal_length_m*1000*estHeight*image_centre_h/((max(heightPx-8,1))*7.32))
    
    return camera_focal_length_m*1000*estHeight*image_centre_h/((max(heightPx-6,1))*8)#(f ) / 
imageiterator = gothroughfiles()

cv2.namedWindow('yeet')
cv2.namedWindow('yeet2')
cv2.namedWindow('yeet3')


clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(32,32))
mask = cv2.imread('yolo/mask.png')
maskbw = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
#fails here 
#TTBB-durham-02-10-17-sub10/left-images/1506943261.483305_L.png

#dp = wlsdisp(maxdisparity)
dp = disparityfinder(maxdisparity)

tracker = tracking()




directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left))




left, right = next (imageiterator)

people = []
while left is not None:
    # leftfname = 'TTBB-durham-02-10-17-sub10/left-images/1506942689.476653_L.png'
    # leftfname = 'TTBB-durham-02-10-17-sub10/left-images/1506943442.478606_L.png'
    # leftoriginal = cv2.imread(leftfname)
    # left = leftoriginal
    # right = cv2.imread(leftfname.replace("_L", "_R").replace('left','right'))
    
    leftoriginal = left
    t = ti()
    classIDs,confidences,boxes = yl.objectsInFrame(leftoriginal[:-120],0.7)
    










    print("yolo time: "+str(ti()-t))
    t = ti()
    # left = cv2.bitwise_and(leftoriginal,mask)
    # right = cv2.bitwise_and(right,mask)
    leftG = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    rightG = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    leftG = cv2.fastNlMeansDenoising(leftG,h=denoise)
    rightG = cv2.fastNlMeansDenoising(rightG,h=denoise)
    # leftG = cv2.medianBlur(leftG,5)
    # rightG = cv2.medianBlur(rightG,5)
    # leftG = cv2.bilateralFilter(leftG,5,20,25)
    # rightG = cv2.bilateralFilter(rightG,5,20,25)









    
    left = clahe.apply(leftG)#cv2.equalizeHist(leftG)
    right = clahe.apply(rightG)#cv2.equalizeHist(rightG)
    # left = np.power(left, 0.75).astype('uint8')
    # right = np.power(right, 0.75).astype('uint8')


    print("preproc time: "+str(ti()-t))
    t=ti()
    
    




    disp, dispscaled,rawdisp = dp.getDisparity(left,right)
    _, fdispscaled, _ = dp.getDisparity(np.flip(right[:,:maxdisparity*2+2],1),np.flip(left[:,:maxdisparity*2+2],1))
    fdispscaled = np.flip(fdispscaled,1)

    dispscaled[:,:maxdisparity] = fdispscaled[:,:maxdisparity]
    pointsmap = np.nan_to_num((399.9745178222656*0.2090607502/dispscaled),nan=0,posinf = 0,neginf=0).astype(np.uint8)
    fpoints = np.nan_to_num((399.9745178222656*0.2090607502/fdispscaled),nan=0,posinf = 0,neginf=0).astype(np.uint8)
    
    # pointsmap[:,:maxdisparity] = fpoints[:,:maxdisparity]

    
    pointsmap = cv2.bitwise_and(pointsmap,maskbw)
    distmap = cv2.bitwise_and(dispscaled,maskbw)
    distmap = cv2.equalizeHist(distmap)
    
    #altpointsmap = dp.threed2twod(points,left.shape[:2])

    
    
    
    
    print("dispr time: "+str(ti()-t))
    t = ti()





    
    distances = [ 0 for i in range(len(boxes))]
    heurdistance = [ 0 for i in range(len(boxes))]
    for o in range(len(boxes)):
        top = max(0,boxes[o][1])
        bottom = max(top+1,boxes[o][3]+boxes[o][1])
        left = max(0,boxes[o][0])
        right = max(left+1,boxes[o][2]+boxes[o][0])
        if boxes[o][3]>1 and boxes[o][2]:
            bounding = np.sort(pointsmap[top:bottom,left:right].flatten())
            # print(bounding.shape,boxes[o][1],boxes[o][3]+boxes[o][1],boxes[o][0],boxes[o][2]+boxes[o][0])
            try:
                
                # distances[o] = (np.mean(np.percentile(bounding,[i for i in range(30,50,1)]))+np.mean(bounding))/2
                distances[o] = np.mean(bounding[int(len(bounding)*1/5):int(len(bounding)*3/5)])
                newdistances = distances[:]
                #heurdistance[o] = distanceFromHeight(classheights[o],abs(top-bottom))
            except:
                pass
            first = np.copy(leftoriginal[top:bottom, left:right])
            # if classIDs[o]==0:#<len(classheights):
    
    #######tracking 
    # itemdict = {}
    # for o in range(len(classIDs)):
    #     if classIDs[o] in itemdict:
    #         itemdict[classIDs[o]].append((boxes[o],distances[o]))
    #     else:
    #         itemdict[classIDs[o]]= [(boxes[o],distances[o])]
    # # print(yeet)
    # preds = tracker.add_frame(itemdict)#classIDs,boxes,distances)
    
    # nexts  = tracker.predict_next()
    # # print(preds)
    # classIDs,boxes,distances,predictions = [],[],[],[]
    # for i in preds:
    #     for o in range(len(preds[i][0])):
    #         classIDs.append(i)
    #         boxes.append(preds[i][0][o])
    #         distances.append(preds[i][1][o])
    #         predictions.append(preds[i][2][o])
    # # print(predictions)
    # # print(boxes)
    # # print(classIDs)
    # # print(distances)
    # newdistances  = [i[2] for i in predictions]
    # # for i in range(len(distances)):
    # #     if newdistances[o]<0.1:
    # #         newdistances[o] = distanceFromHeight(classIDs[o],boxes[o][2])
    # #print(newdistances)
    # for i in range(len(classIDs)):    
    #     cv2.circle(leftoriginal,(int(predictions[i][0]),int(predictions[i][1])),10,(200,200,255),thickness=cv2.FILLED)
    

    print("ranging time: "+str(ti()-t))


    for o in range(len(boxes)):
        yl.drawPred(pointsmap,yl.getName(classIDs[o]),newdistances[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
        yl.drawPred(leftoriginal,yl.getName(classIDs[o]),newdistances[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))


    cv2.imshow('yeet',leftoriginal)    
    cv2.imshow('yeet2',leftG)
    
    cv2.imshow('yeet3',cv2.equalizeHist(pointsmap))
    cv2.waitKey(3)

    # try:
    left,right = next(imageiterator,None)
    # except:
    #     left = None
cv2.destroyAllWindows()
