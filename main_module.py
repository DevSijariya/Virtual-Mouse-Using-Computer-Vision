import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpobject = mp.solutions.hands
hands = mpobject.Hands()
draw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0


while True:
    test , img = cap.read()

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    det_result = hands.process(rgb_img)

    if det_result.multi_hand_landmarks:
        for hand in det_result.multi_hand_landmarks:
            for index,lm in enumerate(hand.landmark):
                height , width , center = img.shape
                x, y = int(width*lm.x) , int(height*lm.y)

                cv2.circle(img, (x,y), 10, (255,0,255), cv2.FILLED)

            draw.draw_landmarks(img,hand,mpobject.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)),(10,90),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),4)

    cv2.imshow("Hand Tracking",img)
    cv2.waitKey(1)
