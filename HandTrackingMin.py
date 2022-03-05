import cv2 as cv
import time 
import mediapipe as mp

# Hand Tracking : Palm Detection Hand Landmarks

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
ptime = 0
ctime = 0
TrackingIds = [4,8,12,16,20]

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x * w) , int(lm.y * h)
                print(id, cx , cy)
                if id in TrackingIds:
                    cv.circle(img,(cx,cy),25,(255,0,255),cv.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)



    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img,str(int(fps)),(18,70),cv.FONT_HERSHEY_COMPLEX,3,(0,0,0),3)

    cv.imshow('Video',img)
    cv.waitKey(1)

cap.release()