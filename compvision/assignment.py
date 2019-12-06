import yolo.yolomodule as yl
#import stereodisparity.disparitymodule as dp
import numpy as np
import cv2
import csv
import os
from time import time as ti
yeet = "yeet" #debugging purposes

master_path_to_dataset = "TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed


full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);


left_file_list = sorted(os.listdir(full_path_directory_left));

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5
maxdisparity = 128

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
        #for each item, using it's previous two frame's positions work out the velocity and predict the next position
        for obj in self.items:
            for i in range(len(self.items[obj])): 
                
                if len(self.items[obj][i])>1:
                    
                    self.items[obj][i].append([2*self.items[obj][i][-1][k]-self.items[obj][i][-2][k] for k in range(3)])
                else:
                    self.items[obj][i].append(self.items[obj][i][-1])
        return self.items
    def add_frame(self,itemdict):
        oldpreds = dict(self.items)#store the old predictions
        self.items = {}#clear out items
        returnsdict = {}#set up a dictionary to return
        for i in itemdict:#for each type of item in the frame
            boxes = list(list(zip(*itemdict[i]))[0])#get the bounding boxes and disparity measured distances
            dists = list(list(zip(*itemdict[i]))[1])
            self.items[i] = []
            returnsdict[i] = [boxes,dists,[]]

            if i not in oldpreds:
                
                oldpreds[i] = []
           
            o = 0
            while o< len(boxes):#while there are more of this class to store/measure/match
                pos = [boxes[o][0]+boxes[o][2]/2,boxes[o][1]+boxes[o][3]/2,dists[o]]#get the measured position
                o+=1
                
                if len(oldpreds[i])>0:#if we have at least one prediction
                    
                    tpos= pos[:]
                    tpos[2]*=7#scale the distance, because it is smaller than the pixel distance
                    diffs = list(enumerate([euclidian(pos,pred[-1][:2]+[pred[-1][2]*7]) for pred in oldpreds[i]]))
                    
                    closest = min(diffs,key= lambda x:x[1])#find the predicted point it was closest to 

                    if closest[1]<pred_limit and pos[2]>0.5:#if it is close enough and our guess was good enough
                        pos = [pos[h]*(1-pred_weight)+oldpreds[i][closest[0]][-1][h]*pred_weight for h in range(len(pos))]#estimate the position using a weighted avg
                    if pos[2]<=0.5:
                        pos[2] = distanceFromHeight(classheights[i],boxes[o-1][3])#naiively estimate the distance using the height of the object

                    tmp = oldpreds[i].pop(closest[0])[:-1]+[pos]#add it to our predictions, removing the complete guess
                    returnsdict[i][2].append(pos)#add it to the thing to return

                    self.items[i].append(tmp)
                else:#if we have no data
                    if pos[2]<0.5:#if disparity was bad
                        pos[2] = distanceFromHeight(classheights[i],boxes[o-1][3])#make a heuristic guess
                    returnsdict[i][2].append(pos)#store new position data
                    self.items[i].append([pos])
        
        return returnsdict#return new positions
            
def euclidian(a,b):
    if len(a)==len(b):
        return sum([abs(a[i]**2-b[i]**2) for i in range(len(a))])**.5
    return None

        


##### class that initialises the disparity processor and has a method for getting the disparity between two images
class disparityfinder:
    def __init__(self,maxdisp = 128):

        self.maxdisparity = maxdisp
        self.stereoProcessor = cv2.StereoSGBM_create(0, maxdisp, 11,20*window_size**2,60*window_size**2)#21)#initialise the processor
        
    def getDisparity(self,left,right,crop_disparity = False):

        originaldisparity = self.stereoProcessor.compute(left,right)#compute disparity

        _, disparity = cv2.threshold(originaldisparity,0, self.maxdisparity * 16, cv2.THRESH_TOZERO)#threshold to make the -1 value 0 and scale
        disparity_scaled = (disparity / 16.).astype(np.uint8)#scale the disparity

        if (crop_disparity):
            width = np.shape(disparity)[1] 
            disparity_scaled = disparity_scaled[0:544,135:width]
        return (disparity_scaled * (256. / self.maxdisparity)).astype(np.uint8), disparity_scaled, originaldisparity#return every version of disparity just incase


# ##### tested using the WLS filter to improve the disparity 
# class wlsdisp:
#     def __init__(self,maxdisp):
        
#         self.maxdisparity = maxdisp
#         self.left_matcher = cv2.StereoSGBM_create(0, maxdisp, 11,20*window_size**2,60*window_size**2)#21)
#         #     minDisparity=0,
#         #     numDisparities=maxdisp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
#         #     blockSize=5,
#         #     P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
#         #     P2=32 * 3 * window_size ** 2,
#         #     disp12MaxDiff=1,
#         #     uniquenessRatio=15,
#         #     speckleWindowSize=0,
#         #     speckleRange=2,
#         #     preFilterCap=63,
#         #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#         # )
        
#         self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
#         self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
#         self.wls_filter.setLambda(lmbda)
#         self.wls_filter.setSigmaColor(sigma)
#     def getDisparity(self,left,right):
#         displ = self.left_matcher.compute(left, right)  # .astype(np.float32)/16
#         dispr = self.right_matcher.compute(right, left)  # .astype(np.float32)/16
#         displ = np.int16(displ)
#         dispr = np.int16(dispr)
#         return 0,(self.wls_filter.filter(displ, left, None, dispr)/16).astype(np.uint8),0


##### an iterator that when called next will give the next two images

def gothroughfiles():
    skipto = ""#"1506943695.479492"#"1506943007.479011"#"1506943441.478664"#"1506943097.480891"#"1506942566.482532"#"1506942553.483915"#"1506942485.480259"
    for filename_left in left_file_list:
        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)
        if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :#as long as both images exist
            if ((len(skipto) > 0) and not(skipto in filename_left)):#if we want to jump ahead
                continue
            elif ((len(skipto) > 0) and (skipto in filename_left)):#if we reach our destination
                skipto = ""

            print(full_path_filename_left)
            print(full_path_filename_right,end='')#leave the endline blank so we can write the min dist later
            # print()
            yield cv2.imread(full_path_filename_left),cv2.imread(full_path_filename_right)#return the two images
def distanceFromHeight(estHeight, heightPx):
    
    
    return camera_focal_length_m*1000*estHeight*image_centre_h/((max(heightPx-6,1))*8)#based on the estimated height and the height in pixels calculate the height




imageiterator = gothroughfiles() #initialise the iterator that runs through the files

left, right = next (imageiterator)#get the first images

cv2.namedWindow('yeet')#create the window for displaying our images


clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(16,16))#initialise the CLAHE processor



mask = cv2.imread('yolo/mask.png')#load the mask that culls the car and top of the screen
maskbw = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)#create a greyscale version



dp = disparityfinder(maxdisparity)#create the disparity processor

tracker = tracking()#instanciate the tracking class








while left is not None:#while we have a left (and therefore right) image
    
    leftoriginal = left#make a copy of the left image for safekeeping
    t = ti()

    
    classIDs,confidences,boxes = yl.objectsInFrame(leftoriginal[:-120],0.7)#use yolo to get all the objects in frame
    


    # print("yolo time: "+str(ti()-t))
    t = ti()

    


    denoise = 5

    # left = cv2.bitwise_and(leftoriginal,mask)
    # right = cv2.bitwise_and(right,mask)
    leftG = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)#make both images greyscale
    rightG = cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    # leftG = cv2.fastNlMeansDenoising(leftG,h=5)
    # rightG = cv2.fastNlMeansDenoising(rightG,h=5)
    # leftG = cv2.medianBlur(leftG,7)
    # rightG = cv2.medianBlur(rightG,7)
    leftG = cv2.bilateralFilter(leftG,5,20,25)#bilateral filter to smooth images while maintaining edges
    rightG = cv2.bilateralFilter(rightG,5,20,25)






    #apply adaptive histogram equalisation using the CLAHE we instanciated earlier

    left = clahe.apply(leftG)#cv2.equalizeHist(leftG)
    right = clahe.apply(rightG)#cv2.equalizeHist(rightG)
    # left = np.power(left, 0.75).astype('uint8')
    # right = np.power(right, 0.75).astype('uint8')


    # print("preproc time: "+str(ti()-t))
    t=ti()
    
    


    #get the disparity values
    
    disp, dispscaled,rawdisp = dp.getDisparity(left,right)

    #get the disparity values for the leftmost part of the image by flipping
    _, fdispscaled, _ = dp.getDisparity(np.flip(right[:,:maxdisparity*2+2],1),np.flip(left[:,:maxdisparity*2+2],1))

    #unflip it
    fdispscaled = np.flip(fdispscaled,1)


    #include the new data
    dispscaled[:,:maxdisparity] = fdispscaled[:,:maxdisparity]

    #turn the disparity map into a depthmap
    pointsmap = np.nan_to_num((399.9745178222656*0.2090607502/dispscaled),nan=0,posinf = 0,neginf=0).astype(np.uint8)
    
    
    pointsmap = cv2.bitwise_and(pointsmap,maskbw)#remove unnecessary stuff (where roof of car was)
    distmap = cv2.bitwise_and(dispscaled,maskbw)
    distmap = cv2.equalizeHist(distmap)#scale for displaying
    
    #altpointsmap = dp.threed2twod(points,left.shape[:2])

    # print("dispr time: "+str(ti()-t))
    t = ti()





    
    distances = [ 0 for i in range(len(boxes))]#make space to store distances
    
    newdistances = distances[:]#make space to store distances measured with heuristics
    
    for o in range(len(boxes)):#for each detected box
        top = max(0,boxes[o][1])
        bottom = max(top+1,boxes[o][3]+boxes[o][1])
        left = max(0,boxes[o][0])
        right = max(left+1,boxes[o][2]+boxes[o][0])
        if boxes[o][3]>1 and boxes[o][2]:
            bounding = np.sort(pointsmap[top:bottom,left:right].flatten())#sort so we can take percentiles
            # print(bounding.shape,boxes[o][1],boxes[o][3]+boxes[o][1],boxes[o][0],boxes[o][2]+boxes[o][0])
            try:
                
                # distances[o] = (np.mean(np.percentile(bounding,[i for i in range(30,50,1)]))+np.mean(bounding))/2
                distances[o] = np.mean(bounding[int(len(bounding)*1/5):int(len(bounding)*3/5)])#take the mean of the 20th to 60th percentile
                # meandistance[o] = np.mean(bounding)
                # meddistance[o] = np.median(bounding)
                # vmid , hmid= (top+bottom)//2,(left+right)//2
                # vh,hh = abs(top-bottom)//4,abs(left-right)//4
                # centremean[o] = np.mean(pointsmap[vmid-vh:vmid+vh,hmid-hh:hmid+hh])


                newdistances = distances[:]

                #heurdistance[o] = distanceFromHeight(classheights[o],abs(top-bottom))
            except:
                pass
            first = np.copy(leftoriginal[top:bottom, left:right])#just for displaying an arbitrary object
            
    
    #######tracking 
    itemdict = {}
    for o in range(len(classIDs)):#split the objects up by classID into a dict
        if classIDs[o] in itemdict:
            itemdict[classIDs[o]].append((boxes[o],distances[o]))
        else:
            itemdict[classIDs[o]]= [(boxes[o],distances[o])]
    # print(yeet)
    preds = tracker.add_frame(itemdict)#add data to the tracker and get returned "better" distance values
    
    nexts  = tracker.predict_next()#make the tracker predict the next frame's values
    # print(preds)
    classIDs,boxes,distances,predictions = [],[],[],[]
    for i in preds:#unwrap the dictionary back into lists
        for o in range(len(preds[i][0])):
            classIDs.append(i)
            boxes.append(preds[i][0][o])
            distances.append(preds[i][1][o])
            predictions.append(preds[i][2][o])

    newdistances  = [i[2] for i in predictions]#get the new Z values 
    

    # print("ranging time: "+str(ti()-t))
    my = np.copy(leftoriginal)#make a copy of the image for drawing on
    #noheur = np.copy(leftoriginal)
    # meds = np.copy(leftoriginal)
    # centr = np.copy(leftoriginal)
    for i in range(len(classIDs)): #draw predicted improved positions as circles  
        cv2.circle(my,(int(predictions[i][0]),int(predictions[i][1])),10,(200,200,255),thickness=cv2.FILLED)
    for o in range(len(boxes)):
        #yl.drawPred(pointsmap,yl.getName(classIDs[o]),newdistances[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
        yl.drawPred(my,yl.getName(classIDs[o]),newdistances[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
        #yl.drawPred(noheur,yl.getName(classIDs[o]),distances[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
        # yl.drawPred(meds,yl.getName(classIDs[o]),meddistance[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
        # yl.drawPred(centr,yl.getName(classIDs[o]),centremean[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))

    scale_percent = 80 # percent of original sizenp.zeros(leftoriginal.shape).astype(np.uint8)
    #cv2.cvtColor(clahe.apply(cv2.cvtColor(leftoriginal,cv2.COLOR_BGR2GRAY)),cv2.COLOR_GRAY2BGR)
    todisp = np.concatenate([my,cv2.cvtColor(cv2.bitwise_and(dispscaled,maskbw),cv2.COLOR_GRAY2BGR)],axis = 0)#join the two images we are displaying together
    # todisp = np.concatenate([todisp,np.concatenate([meds,centr],axis= 1)],axis = 0)
    
    # resize image
    todisp = cv2.resize(todisp, (int(todisp.shape[1] * scale_percent / 100),int(todisp.shape[0] * scale_percent / 100)), interpolation = cv2.INTER_AREA)#scale them
    # print(todisp.shape)
    # if countdown>0:
    #     output.write(todisp)
        
    # else:
    #     while countdown > -270:
    #         countdown-=1
    #         try:
    #             left,right = next(imageiterator)
    #         except:
    #             left = None
    #     countdown = 30
    # countdown-=1
    # print(countdown)
    if len(distances)>0:#if there is an object in scene
        print(' : nearest detected scene object ({0}m)'.format(min(distances)))
    else:
        print()
    
    cv2.imshow('yeet',todisp) #display our images
    
    
    cv2.waitKey(5)#wait up to 5ms to see if the user presses a key

    try:
        left,right = next(imageiterator)#try and get the next image, will error if none exists
    except:
        left = None
cv2.destroyAllWindows()#close all windows
# output.release()
