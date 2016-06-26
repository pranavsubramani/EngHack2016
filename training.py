import cv2
import numpy as np
from PIL import Image

#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

############################# testing part  #########################

im = cv2.imread('sudoku.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

width, height = im.shape[:2]
print "Width and height is: " + str(width) + "x" + str(height)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

def findPos(x, y, width, height):
    retX, retY = 0, 0
    for i in range(0, 9):
        if(x >= i*width and x <= (i+1)*width):
            retX = i
        if(y >= i*height and y <= (i+1)*height):
            retY = i;
    return retX, retY
    
for cnt in contours:
    if cv2.contourArea(cnt)>150 and cv2.contourArea(cnt)<1500:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h<40:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            ronew = gray[y:y+h,x:x+w]
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            retX, retY = findPos(x, y, width/9, height/9)
            if(retX == 0):
                imageName = str(retY) + "image"
                cv2.imshow('imageName', ronew)
                cv2.waitKey(0)
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

cv2.imshow('im',im)
cv2.imshow('gray', gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
