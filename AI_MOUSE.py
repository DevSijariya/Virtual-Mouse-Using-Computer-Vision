import cv2,time,autopy
import numpy as np
import tracking_module as tm

width,height = 640,480
frameR = 100 
smoothening = 10


wScr, hScr = autopy.screen.size()
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

obj = tm.HandTracking()

prev_time = 0

while True:
    test, img = cap.read()
    img = obj.hand_tracker(img)
    pos_list , bound_box = obj.finger_tracker(img,box=True,show=False)

    if len(pos_list)!=0:
        x1,y1 = pos_list[8][1:]
        x2,y2 = pos_list[12][1:]

        fingers = obj.decision()
        cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR),(255, 0, 255), 2)
        
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, width - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, height - frameR), (0, hScr))
            
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
        
           
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        

        if fingers[1] == 1 and fingers[2] == 1:
            
            length, img, lineInfo = obj.distance(8, 12,img)
            
        
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img,str(int(fps)),(10,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Virtual Mouse",img)
    cv2.waitKey(1)