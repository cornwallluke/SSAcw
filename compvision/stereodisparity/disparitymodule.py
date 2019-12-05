#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################
####usage import disparitymodule
####      disparitymodule.getDisparity(leftImage, rightImage) returns disparity map image
import cv2
import os
import numpy as np

# where is the data ? - set this to where you have it

master_path_to_dataset = "../TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = "" # set to timestamp to skip forward to

crop_disparity = False # display full or cropped disparity image
pause_playback = False # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)



# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21)

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)#21)

def getDisparity(imgL,imgR):
    

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    originaldisparity = stereoProcessor.compute(imgL,imgR)

    # filter out noise and speckles (adjust parameters as needed)

    # dispNoiseFilter = 5 # increase for more agressive filtering
    # cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(originaldisparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    # crop disparity to chop out left part where there are with no disparity
    # as this area is not seen by both cameras and also
    # chop out the bottom area (where we see the front of car bonnet)

    if (crop_disparity):
        width = np.shape(disparity)[1] #np.size(disparity_scaled, 1)
        disparity_scaled = disparity_scaled[0:544,135:width]
    return (disparity_scaled * (256. / max_disparity)).astype(np.uint8), disparity_scaled, originaldisparity
    
#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, rgb=[]):

    points = []
    pointsmap = np.zeros((disparity.shape),dtype = np.uint8)

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]
    #print(height,width)

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = .... Y = ... below

    # Zmax = ((f * B) / 2)

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x]
                pointsmap[y,x]= Z
                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points

                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]])
                else:
                    points.append([X,Y,Z])
                    np.nan_to_num()
    return points,pointsmap

#####################################################################

# project a set of 3D points back the 2D image domain

def threed2twod(points,dimensions):

    points2 = np.zeros(dimensions).astype(np.uint8)

    # calc. Zmax as per above

    # Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2

    for i in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again
        if points[i][2] > 0:
            x = ((points[i][0] * camera_focal_length_px) / points[i][2]) + image_centre_w
            y = ((points[i][1] * camera_focal_length_px) / points[i][2]) + image_centre_h
            #print(points[i][2])
            points2[int(y),int(x)]=int(points[i][2])
            

    return points2

#####################################################################


if __name__=='__main__':
    # get a list of the left image files and sort them (by timestamp in filename)

    left_file_list = sorted(os.listdir(full_path_directory_left))
    for filename_left in left_file_list:

        # skip forward to start a file we specify by timestamp (if this is set)

        if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
            continue
        elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
            skip_forward_file_pattern = ""

        # from the left image filename get the correspondoning right image

        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
        full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

        # for sanity print out these filenames

        print(full_path_filename_left)
        print(full_path_filename_right)
        print()

        # check the file is a PNG file (left) and check a correspondoning right image
        # actually exists

        if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

            # read left and right images and display in windows
            # N.B. despite one being grayscale both are in fact stored as 3-channel
            # RGB images so load both as such

            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            cv2.imshow('left image',imgL)

            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
            cv2.imshow('right image',imgR)

            print("-- files loaded successfully")
            print()

            
            # perform preprocessing - raise to the power, as this subjectively appears
            # to improve subsequent disparity calculation

            imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

            imgL = np.power(imgL, 0.75).astype('uint8')
            imgR = np.power(imgR, 0.75).astype('uint8')


            cv2.imshow("disparity",getDisparity(imgL,imgR).astype(np.uint8))

            # keyboard input for exit (as standard), save disparity and cropping
            # exit - x
            # save - s
            # crop - c
            # pause - space

            key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
            if (key == ord('x')):       # exit
                break # exit
            elif (key == ord('s')):     # save
                cv2.imwrite("sgbm-disparty.png", disparity_scaled)
                cv2.imwrite("left.png", imgL)
                cv2.imwrite("right.png", imgR)
            elif (key == ord('c')):     # crop
                crop_disparity = not(crop_disparity)
            elif (key == ord(' ')):     # pause (on next frame)
                pause_playback = not(pause_playback)
        else:
                print("-- files skipped (perhaps one is missing or not PNG)")
                print()

    # close all windows

    cv2.destroyAllWindows()

    #####################################################################
