import time
import autopy
import cv2 as cv
import mediapipe as mp
import numpy as np
import math

cap = cv.VideoCapture(0)
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap.set(3, wCam)
cap.set(4, hCam)
wScreen, hScreen = autopy.screen.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1)
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

    smoothening = 7
    pLocX, pLocY = 0, 0
    cLocX, cLocY = 0, 0
    
    while True:
        success, img = cap.read()
        img = findHands(img)
        lmList = findPosition(img)

        if len(lmList) != 0:

            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            cv.rectangle(img, (frameR, 10), (wCam-frameR, hCam-190), (255, 255, 0), 2)

            if(lmList[8][2] < lmList[5][2] and lmList[12][2] > lmList[9][2]):

                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
                y3 = np.interp(y1, (frameR, hCam-190), (0, hScreen))

                cLocX = pLocX + (x3 - pLocX) / smoothening
                cLocY = pLocY + (y3 - pLocY) / smoothening

                autopy.mouse.move(wScreen - cLocX, cLocY)
                pLocX, pLocY = cLocX, cLocY 

            if(lmList[8][2] < lmList[5][2] and lmList[12][2] < lmList[9][2]):
                length = math.hypot(x2 -  x1, y2 - y1)

                if length < 35:
                    autopy.mouse.click()

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
