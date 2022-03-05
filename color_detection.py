import cv2
import argparse

#function to get x,y coordinates of mouse double click
def find_color( x,y,img):
    global b,g,r
    b,g,r = img[y,x]
    b = int(b)
    g = int(g)
    r = int(r)

def fead_video(img_path):
    vid = cv2.VideoCapture(int(img_path))
    while(True):
        x=40
        y =10
        ret,frame = vid.read()
        find_color(x,y,frame)
        frame = cv2.circle(frame,(x,y),radius=0,color=(r,g,b),thickness=1)
        frame = cv2.rectangle(frame,(x,y),(x+10,y+10),color=(123,32,44),thickness=1)\
        
        cv2.imshow('video',frame)
        
        print(b,g,r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

def image_fead(img_path):
    img = cv2.imread(img_path)
    #declaring global variables (are used later on)
    r = g = b = xpos = ypos = 0
    while(1):
        cv2.imshow("image",img)
        #Break the loop when user hits 'esc' key    
        if cv2.waitKey(20) & 0xFF ==27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #Creating argument parser to take image path from command line
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image','--source', required=True, help="Image Path")
    args = vars(ap.parse_args())
    img_path = args['image']
    fead_type = 0
    if(img_path=="0"):
        fead_type = 1

    print(fead_type)

    if(fead_type==0):
        image_fead(img_path)
    else :
        fead_video(img_path)


