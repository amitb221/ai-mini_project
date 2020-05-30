
import cv2 
import numpy as np 
import matplotlib.pyplot as plt


def canny_edge_detector(image): 
    #converting image in gray color

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
 
    #soomthing image to reduce noise   
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    #Detecting edges in the images   
    canny = cv2.Canny(blur, 50, 150) 
    return canny 


def region_of_interest(image): 
    height = image.shape[0] 
    polygons = np.array([ [(200, height), (1100, height), (550, 250)] ]) 
    mask = np.zeros_like(image)  
    #cv2.fillPoly fills the area bounded by one polygon.
    cv2.fillPoly(mask, polygons, 255)  
    #using cv2.bitwise_and to mask the iamge
    masked_image = cv2.bitwise_and(image, mask)  
    return masked_image 

#getting cordinates on image
def create_coordinates(image, line_parameters):
    slope,intercept=1,1
    global p,q
    try:
        slope, intercept = line_parameters 
        p,q=slope,intercept
    except TypeError:
        slope,intercept=p,q
    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 

#creating lines from coordinate on images
def average_slope_intercept(image, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept))               
    left_fit_average = np.average(left_fit, axis = 0) 
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = create_coordinates(image, left_fit_average) 
    right_line = create_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line]) 


def display_lines(image, lines): 
    line_image = np.zeros_like(image) 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            #drawing the lane-line on the image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 15 ) 
    return line_image 



#Reading the video input file
cap = cv2.VideoCapture("/home/amit/Documents/test2.mp4")  
while(cap.isOpened()): 
    #capturing the frame
    ret, frame = cap.read()
    #Soomthing and creating canny edge image
    canny_image = canny_edge_detector(frame) 
    #creatiing region of intrest
    cropped_image = region_of_interest(canny_image) 
    #using Probablistic Hough tranform to find lane lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,  
                            np.array([]), minLineLength = 40,  
                            maxLineGap = 5)  
    #
    averaged_lines = average_slope_intercept(frame, lines)  
    #Drwaing lane line
    line_image = display_lines(frame, averaged_lines)
    #combining original image and lane line detected image 
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  
    #displaying final image
    cv2.imshow("results",combo_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):       
        break
cap.release()  
cv2.destroyAllWindows()  