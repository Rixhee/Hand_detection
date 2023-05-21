import cv2 as cv
import numpy as np
import time
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def findHands(img):
    global results 
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img

def findPosition(img, handNo = 0, draw=True):

    lmList = []

    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            if draw:
                lmList.append([id, cx, cy])

    return lmList

def main():
    
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        img = findHands(img)
        lmList = findPosition(img)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
