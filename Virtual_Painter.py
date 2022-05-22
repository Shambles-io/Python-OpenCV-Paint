import cv2
import numpy as np
import os
import handTrackingModule as htm

# Importing images
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = [] # List to store all imaged we want to overlay

for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}') # Complete path to read images from
    overlayList.append(image) # Appending our images to the overlay list

print(len(overlayList)) # Checking to make sure we imported all images (5)

# Run our webcam, when we hae webcam video, we can overlay one of the images by default
header = overlayList[0]

# Color codes
drawColor = (255, 0, 255)
# Brush thickness
brushThickness = 15
eraserThickness = 45


# Camera settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Set capture window width
cap.set(4, 720) # Set capture window height


detector = htm.handDetector(detectionCon=0.75, maxHands=1) # Using HTM hand detector
xp, yp = 0, 0 # previous x and y points
imgCanvas = np.zeros((720, 1280, 3), np.uint8) # Create image camvas with numpy

# Loop to run webcam
while True:
    # 1. IMPORT CAMERA IMAGE
    success, img = cap.read()
    img = cv2.flip(img, 1) # We need to flip the image horizontally to mirror ourselves

    # 2. FINDING HAND LANDMARKS
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) # List of hand landmarks

    # Checking length of landmark list
    if len(lmList) != 0:
        fingers = [] # Create list for our fingers
        #print(lmList)

        # Getting position of index finger tip
        x1, y1 = lmList[8][1:] # 8 is the landmark for index fingertip

        # Getting position of middle finger tip
        x2, y2 = lmList[12][1:] # 12 is the landmark for middle fingertip


        # 3. CHECK WHICH FINGERS ARE UP
        fingers = detector.fingersUp()
        #print(fingers)
        
        # 4. IF SELECTION MODE - 2 FINGERS UP
        if fingers[1] and fingers[2]:
            # # Visual indicater of Selection Mode by drawing a rectangle that connects index and middle finger tips
            # cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)
            xp, yp = 0, 0
            # print("Selection Mode")
            # Checking for the click
            if y1 < 125: # Area within the header image
                if 112 < x1 < 277: 
                    header = overlayList[0]
                    drawColor = (0, 0, 255) # RED
            
                elif 360 < x1 < 525: 
                    header = overlayList[1]
                    drawColor = (171, 71, 0) # COBALT BLUE
                
                elif 610 < x1 < 773: 
                    header = overlayList[2]
                    drawColor = (0, 255, 0) # GREEN

                ################ CURRENTLY DISABLING UNTIL I CAN WORKOUT SOME ISSUES
                elif 868 < x1 < 1032: 
                    header = overlayList[3]
                    drawColor = (255, 255, 255) # WHITE
                ##################

                elif 1100 < x1 < 1186:
                    header = overlayList[4]
                    drawColor = (0, 0, 0) # ERASER / BLACK

            # Visual indicater of Selection Mode by drawing a rectangle that connects index and middle finger tips
            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)

        # 5. IF DRAWING MODE - 1 FINGER UP
        if fingers[1] and fingers[2] == False:
            # Visual indicater of Drawing Mode by drawing a circle over index tip
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            # print("Drawing Mode")

            # At the first iteration/frame, xp and yp = 0, which would cause a line to draw from the 0,0 position to the users point, which isn't good
            if xp == 0 and yp == 0: # Very first frame we detect hand or start to draw
                xp, yp = x1, y1 # Now, whatever value we are at, we want to draw at the same point


            # Increase eraser size for easier use
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1 # continues to keep updating previous points to current points

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY) # Convert imgCanvas from bgr to grayscale
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV) # Convert gray image to an image inverse (black -> white, white (or any color) -> black)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse) # Adding inverse image and our webcam image - allowing us to draw (BLACK) on our capture window
    img = cv2.bitwise_or(img, imgCanvas) # Adding image and image canvas - allowing us to draw WITH COLOR on our capture window

    # Setting header image
    img[0:125, 0:1280] = header # We define the headers height is from 0 to 125 and width from 0 to 1280

    # Combining img and imgCanvas and blending them together so we can draw on our image
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) == ord('q'): # If user pressed 'q' the program ends
        break