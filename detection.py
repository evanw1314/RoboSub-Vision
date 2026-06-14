import cv2
import numpy as np

# VideoCapture restricts and begins to use the camera for this script.
# The number passed in as the argument is the camera number of the computer.
cap = cv2.VideoCapture(0)

# isOpened tells us whether the camera is ready to read or not.
if not cap.isOpened():
    print("Camera Error")
    exit()

print("Press 'Q' to quit.")

while True:

    # .read() grabs the next frame from the camera.
    # ret (return value) tells us whether the frame was successfully grabbed or not
    # frame is the next frame from the camera.
    ret, frame = cap.read()

    # If the frame was not successfully grabbed:
    if not ret:
        print("Error: Failed to grab frame.")
        break

    #Convert the image to HSV
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #set boundaries for lower hue values of red
    lower_red1 = np.array([0, 120, 50])
    upper_red1 = np.array([10, 255, 255])

    #set boundaries for higher hue values of red
    lower_red2 = np.array([170, 120, 50])
    upper_red2 = np.array([180, 255, 255])

    #inRange returns an array of values: 0 (Black) if the HSV of the original image falls outside our boundaries, 255 (White) if the HSV of the original image lies inside our boundaries
    mask1 = cv2.inRange(hsvImg, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsvImg, lower_red2, upper_red2)

    #bitwise_or allows us to combine mask1 and mask2 into a complete mask.
    combined = cv2.bitwise_or(mask1, mask2, mask=None)

    #Kernel = size of the brush (area) used for erosion and dilation.
    kernel = np.ones((5, 5), dtype=np.uint8)

    #morphologyEx is a function that allows us to use morphology transformations on an image.
    red_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    #findContours is a function that detects the boundaries of objects in a binary (0 / 255) image.
    #Retrieval types: External = returns outermost contours, ignores inner ones (like concentric circles). List = returns all the contours without hierarchy. Completed Component (CComp) = returns all contours and sorts them into outer / inner contours (kinda complicated). Tree = Calculates the hierarchy of the contours.
    #Chain approximation types (differs on memory usage): None = Stores every single pixel coordinate of the boundary (Big memory usage). Simple = Compresses line segments (leaving only their endpoints), resulting in low memory usage.
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()

    for contour in contours:

        #contourArea function calculates the area of the contour in squared pixels
        area = cv2.contourArea(contour)
        
        #Only draw bounding circles around contours with significant area
        if area > 500:

            #Find the smallest possible circle that completely encloses the contour. Returns the center of the circle as well as its radius.
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)

            #Pixels must be whole numbers, so convert it here.
            center = (int(center_x), int(center_y)) 
            radius = int(radius)

            #Draw a circle (white) on the image where the red region is (editing the image in-place)
            cv2.circle(output, center, radius, (255, 255, 255), 3)

            #An f in front of quotations turns a string into a formatted string: allows variables to be used inside the string
            label = f"Red Area: {int(area)} squared pixels"
            cv2.putText(output, label, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #imshow opens the image in a new window
    #cv2.imshow("Original", frame) 
    cv2.imshow("Masked Image", combined)
    cv2.imshow("Color Detection", output) 

    # waitKey forces the program to wait 1 ms before switching frames (vid), and also returns the ASCII value of key pressed during that time frame.
    # We use & 0xFF (bitwise AND) in order to retrieve the actual ASCII value
    # ord gives the ASCII value of the letter passed in.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'Q' pressed, exiting.")
        break

# Unlocks the camera for other applications to use
cap.release()

# Completely destroys the windows the script created. Without it, the windows could end up being unresponsive.
cv2.destroyAllWindows()