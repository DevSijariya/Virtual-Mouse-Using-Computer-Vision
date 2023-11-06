import cv2,time,math
import mediapipe as mp

class HandTracking():

    def __init__(self):
        self.mpobject = mp.solutions.hands
        self.hands = self.mpobject.Hands(min_detection_confidence=0.7)
        self.draw = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]
        # self.pos_list=[]

    def hand_tracker(self,img,show=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.det_result = self.hands.process(rgb_img)

        if self.det_result.multi_hand_landmarks:
            for hand in self.det_result.multi_hand_landmarks:
                if show:
                    self.draw.draw_landmarks(img,hand,self.mpobject.HAND_CONNECTIONS)
        return img

    def finger_tracker(self,img,show=True,box=False):
        xlist = []
        ylist = []
        bounding = []
        self.pos_list=[]
        if self.det_result.multi_hand_landmarks:
            for hand in self.det_result.multi_hand_landmarks:
                for index,lm in enumerate(hand.landmark):
                        height , width , center = img.shape
                        x, y = int(width*lm.x) , int(height*lm.y)
                        xlist.append(x)
                        ylist.append(y)
                        self.pos_list.append([index, x, y])
                        if show:
                            cv2.circle(img, (x,y), 10, (0,0,255), cv2.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bounding = xmin, ymin, xmax, ymax 
            if box:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)

        return self.pos_list , bounding

    def finger_tracker_vc(self,img,show=True):
        self.pos_list = []
        xlist = []
        ylist = []
    

        if self.det_result.multi_hand_landmarks:
            for hand in self.det_result.multi_hand_landmarks:
                for index,lm in enumerate(hand.landmark):
                        height , width , center = img.shape
                        x, y = int(width*lm.x) , int(height*lm.y)
                        xlist.append(x)
                        ylist.append(y)
                        self.pos_list.append([index, x, y])
                if show:
                    cv2.circle(img, (x,y), 10, (0,0,255), cv2.FILLED)
        return self.pos_list

    def decision(self):
        fingers = []
        if self.pos_list[self.tip[0]][1] > self.pos_list[self.tip[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.pos_list[self.tip[id]][2] < self.pos_list[self.tip[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.pos_list[p1][1:]
        x2, y2 = self.pos_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            l = math.hypot(x2 - x1, y2 - y1)

        return l, img, [x1, y1, x2, y2, cx, cy]


def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    obj = HandTracking()


    while True:
        test , img = cap.read()

        img = obj.hand_tracker(img)
        pos_list = obj.finger_tracker(img)

        # if len(pos_list)!=0:
        #     print(pos_list[4])

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        cv2.putText(img, str(int(fps)),(10,90),cv2.FONT_HERSHEY_PLAIN,4,(255,0,0),4)

        cv2.imshow("Hand Tracking",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
