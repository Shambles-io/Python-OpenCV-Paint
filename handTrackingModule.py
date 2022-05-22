import cv2
import mediapipe as mp
import time
import math


# Creating our class
class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            # handLmrks is a single hand
            for handLmrks in self.results.multi_hand_landmarks:
                if draw:
                    # Here we are doing to draw on a single hand (not the RGB image)
                    self.mpDraw.draw_landmarks(
                        img, handLmrks, self.mpHands.HAND_CONNECTIONS)
        # Returns our "drawn on" image
        return img

    def findPosition(self, img, handNum=0, draw=True):
        # List of landmark positions
        self.lmList = []
        if self.results.multi_hand_landmarks:
            # Here it is getting the first hand "element"
            myHand = self.results.multi_hand_landmarks[handNum]
            # id is the exact index number of the finger landmarks
            # lm is the landmark from .landmarks
            for id, lm in enumerate(myHand.landmark):
                # Each landmark will have a specific x, y, z location
                # print(id, lm)
                HEIGHT, WIDTH, CHANNELS = img.shape
                # Center x and y position
                cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                # Prints each landmarks id, along with the centerx and y location
                #print(id, cx, cy)

                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                # if id = 0 (frst landmark), we will draw a circle at center position
                # If we remove the if statement, it would just draw circles over ALL the landmarks
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 10,
                #                (255, 0, 255), cv2.FILLED)

        return self.lmList


    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    
            # totalFingers = fingers.count(1)
    
        return fingers
 
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)
    
        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
    # Writing the FPS
    prevTime = 0
    currTime = 0
    # Initialize use of integrated webcam
    cap = cv2.VideoCapture(0)
    # Create object for handDetector class
    detector = handDetector()

    while True:
        success, img = cap.read()

        # When we get our img (above line), we will send the image to the detector to find the hands
        img = detector.findHands(img)

        # Calls findPosion method
        # Returns the value of yuor list at any position
        # If we set draw=False, we will not draw the extre purple curcles over our landmarks
        lmList = detector.findPosition(img, draw=False)
        # if len(lmList) != 0:
        #     # Within [] we can enter 0-20, which are the hand landmarks
        #     # Then we will only print the positioon information of that specific landmark
        #     # 4 = Tip of thumb
        #     print(lmList[4])

        # Establishing the fps math
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        # Display the fps on the screen
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
