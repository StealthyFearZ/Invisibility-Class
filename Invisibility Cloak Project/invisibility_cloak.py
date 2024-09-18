import cv2
import time
import numpy as np

#To save the output in a file output.avi
# fourcc is a code to specify the video codec. Codecs are used to compress data.
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#Starting the webcam
cap = cv2.VideoCapture(0)

#Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#Capturing background for 60 frames
for i in range(60):
    ret, bg = cap.read()
#Flipping the background
bg = np.flip(bg, axis=1)

#Reading the captured frame until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    img = np.flip(img, axis=1)

    #Converting the color from BGR to HSV

    # HSV- (hue, saturation, value)
    # hue- This channel encpdes color information
    #Hue is measured in degrees from 0 to 360 along the circumference of the base of the cylinder
    #Saturation- This encodes the intensity of the color, the radius of the cylinder represents saturation
    #Value- This encodes the brightness of the color, shading or gloss components of an image. The height represents value.
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Generating mask to detect red colour(values can be changed)
    #The color red lies between 0 and 10 degrees or 170 to 180 degress(referring to Hue) in HSV
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255,255])
    mask_1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)
    #In the in-range function,
    #  the first parameter that we are passing is the source array, in which we have to compare to check whether it's in a certain range
    #  upper bound array is the array consisting the upper bound, vice versa for lower bound array
    mask_1 = mask_1 + mask_2

    cv2.imshow("mask_1", mask_1)

    #Open and expand the image where there is mask 1 (color)
    mask_1= cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1= cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    #Selecting only the part that does not have mask one and saving in mask 2
    mask_2=cv2.bitwise_not(mask_1)

    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    res_1=cv2.bitwise_and(img,img,mask=mask_2)

    #Keeping only the part of the images with the red color
    res_2=cv2.bitwise_and(bg,bg,mask=mask_1)

    #Generating the final output
    final_output = cv2.addWeighted(res_1,1,res_2,1,gamma=0)
    output_file.write(final_output)
    
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()
#out.release()
cv2.destroyAllWindows()
