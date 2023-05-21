import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import math
from ctypes import cast, POINTER
import comtypes
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interference = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interference, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

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
        
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            cv.circle(img, (x1, y1), 15, (255, 255, 0), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 255, 0), cv.FILLED)
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 0, 0), 3)
            cv.circle(img, (cx, cy), 15, (255, 255, 0), cv.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [30, 180], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            if length < 30:
                cv.circle(img, (cx, cy), 15, (255, 255, 255), cv.FILLED)

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
