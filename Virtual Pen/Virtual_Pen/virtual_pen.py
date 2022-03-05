import cv2 as cv
from cv2 import line
import numpy as np
import os
import time
import HandTrackingModule as htm

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.zeros((720,1280, 3),np.uint8)

while True:
    success, img = cap.read()
    img = cv.flip(img,1)

    # Find Hands

    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) != 0:
        # print(lmlist)

        # Tip of the different fingers
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        x3,y3 = lmlist[16][1:]
        x4,y4 = lmlist[20][1:]
        x5,y5 = lmlist[4][1:]

    # Check weather fingers are up
        fingers = detector.fingersUp()
        print(fingers)

    # 4 if selection mode - two finger are up:
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            cv.rectangle(img,(x1,y1-25),(x2,y2-25),(255,0,255),cv.FILLED)
            print("Selection mode")


    # 5 Drawing Mode
        if fingers[1] and fingers[2]==False:
            cv.circle(img,(x1,y1),10,(255,0,255),cv.FILLED)
            
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            cv.line(img,(xp,yp),(x1,y1),(255,0,0),15)
            cv.line(imgCanvas,(xp,yp),(x1,y1),(255,0,0),15)

            xp,yp = x1,y1

    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)

    img = cv.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv.imshow('Image',img)
    cv.imshow('ImageCanvas',imgCanvas)
    cv.waitKey(1)