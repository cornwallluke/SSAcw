import yolomodule as yl
import cv2
frame = cv2.imread('test2.png')
mask = cv2.imread('mask.png')
frame = cv2.bitwise_and(frame,mask)
classIDs,confidences,boxes = yl.objectsInFrame(frame)
for o in range(len(boxes)):
    yl.drawPred(frame,str(classIDs[o]),confidences[o],boxes[o][0],boxes[o][1],boxes[o][2]+boxes[o][0],boxes[o][3]+boxes[o][1],(255, 178, 50))
cv2.namedWindow('yeet')
cv2.imshow('yeet',frame)
cv2.waitKey()
cv2.destroyAllWindows()
