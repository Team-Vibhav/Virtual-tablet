import cv2
import mediapipe as mp
import time
import math

class handDetector():
   
    def __init__(self, mode = False, maxHands = 1,modelComplexity=1,detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    # cv2.putText(img,)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        self.lmlist = []
      
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # print(myHand.landmark[4].z,myHand.landmark[8].z)
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy,lm.z])
                if draw:
                    cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,255),1)
                    cv2.circle(img, (cx, cy), 3, (255,0,255), cv2.FILLED,cv2.LINE_AA)
            
        return self.lmlist
    def distance (self,p1,p2):
        dis = ((p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5
        centroid = [(p1[1]+p2[1])/2,(p1[2]+p2[2])/2]
        return dis,centroid

    def fingersUp(self): #checking which finger is open 
      
      fingers = [] #storing final result
      # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign
      dis,centroid  = self.distance(self.lmlist[4],self.lmlist[8])
      y = (((1+self.lmlist[4][3])*100)%10)
      prod = dis*y
      print(dis,y)
      if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:#checking x position of 4 is in right to x position of 3
          fingers.append(1)
      else:
          fingers.append(0)

      # Fingers
      for id in range(1, 5):#checking tip point is below tippoint-2 (only in Y direction)
          if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
              fingers.append(1)
          else:
              fingers.append(0)

          # totalFingers = fingers.count(1)

      return fingers,centroid

     
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()