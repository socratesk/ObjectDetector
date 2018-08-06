# Tennis Ball Tracker

# Import necessary libraries
from collections import deque
import cv2
import numpy as np

# Specify range of Tennis Ball color in HSV
yellowLower = np.array([25, 75, 85])   # 28, 82, 80
yellowUpper = np.array([50, 220, 255]) # 45, 204, 255

# Create empty points array deque. The macimum length determines the trail the program leaves
points = deque(maxlen=30)

# Initalize webcam. 0 starts built-in camera
cap = cv2.VideoCapture(0)

# Read captured webcam frame. This is required to get the frame size
ret, frame = cap.read()

# Get default video frame's size
Height, Width = frame.shape[:2]

# Initialize counter. Will be used when the detected tennis ball goes out of webcam
frame_count = 0

while True:

    # Read captured webcam frame
    ret, frame = cap.read()
    
    # Convert color frame into HSV
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv_img, yellowLower, yellowUpper)

        
    # Find contours from HSV masked image
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty centre array to store centroid center of mass
    center = int(Height/2), int(Width/2)

    if len(contours) > 0:
        
        # Get the largest contour and its center 
        c = max(contours, key=cv2.contourArea)
        
        # Extract radius and x & y coordinates of inscribed circle of the contour 
        (x, y), radius = cv2.minEnclosingCircle(c)
        
        # Extract moment points (weighted average of the image pixels' intensities) of image
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except:
            center = int(Height/2), int(Width/2)

        # Allow only countors that have a larger than 25 pixel radius
        if radius > 25:
            
            # Draw cirlce and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
    # Add center points to array deque
    points.appendleft(center)
    
    # loop over the set of tracked points
    if radius > 25:
        
        
        for i in range(1, len(points)):
            
            if points[i - 1] is None or points[i] is None:
                continue
                
            try:
                thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), thickness)
            except:
                pass
            
        # Make frame count zero
        frame_count = 0
    else:
        # Count frames 
        frame_count += 1
        
        # If we count 10 frames without object lets delete our trail
        if frame_count == 10:
            
            # when frame_count reaches 10 let's clear our trail and reset points captured
            frame_count = 0
            points = deque(maxlen=30)
            
     # Flip image so that it aligns with true movement of the ball
    frame = cv2.flip(frame, 1)
    
    # Display our object tracker
    cv2.imshow("Tennis Ball Tracker", frame)
    
    if cv2.waitKey(1) == 13:  # key == 13 or key == ord("q") or ord("Q"):
        break

# Release webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
