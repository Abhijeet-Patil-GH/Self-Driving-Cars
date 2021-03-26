# Cars, Pedestrians and Lanes Detection program for Tesla Self Driving Cars
#   The cars, peds in this program are detected using python opencv library which provides a class called CascadeClassifier which is used to rapidly detect objects from trained data using 'Haar Feature' on GreyScaled images/frames to reduce complexity.
#   The lanes in this program are detected using python opencv library which provides a class called Canny which converts a image/frame into a canny image to reduce complexity and futher the image used to detect only the region of interest.



from cv2 import cv2
import numpy as np 



video = cv2.VideoCapture("tesla_vehicle_dashcam.mp4")                                                             # Select/Capture the Video


cars_detector_file = r"F:\! PortFolio\Python Projects\Self Driving Car\cars_detector.xml"                         # Pre-Trained Data Car,Peds Detection using 'Haar Feature' in xml file
pedestrians_detector_file = r"F:\! PortFolio\Python Projects\Self Driving Car\pedestrians_detector.xml"

cars_detector = cv2.CascadeClassifier(cars_detector_file)                                                         # CascadeClassifier opencv method for object detection using Haar Feature
pedestrians_detector = cv2.CascadeClassifier(pedestrians_detector_file)


# Function to detect car and pedestrians over an image/frame

def car_peds_detection(image):
    grayscaled_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured_frame = cv2.GaussianBlur(grayscaled_frame, (5, 5), 0)

    cars = cars_detector.detectMultiScale(blured_frame)                                                           # Opencv method to detect the objects from CascadeClassifier
    pedestrians = pedestrians_detector.detectMultiScale(blured_frame)

    for (x, y, w, h) in cars :                                                                                    # Create a rectangle provieded by opencv over the obtained co-ordinates from detected cars and peds using Haar Feature of respective length and height 
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians :
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    return image    

# Function to convert an image/frame into canny image/frame

def canny(image) :
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(grayscaled, (5, 5), 0)                                                              # (5, 5) table of convolution for adjusting intense edges into blur with deviation or difference of 0
    canny = cv2.Canny(blured, 50, 150)                                                                            # 50, 150 is the color intensity change in an image which is normally(0 to 225) so 50, 150 is a descent one to a sharp change in the intensity
    return canny

# Function to get only the region of interest over an canny image for detecting the lanes and not the other scenes

def region_of_interest(image) :
    height = image.shape[0]
    polygons = np.array([                                                                                         # Generally the lanes from an FFP view forms a polygon shapes stored in numpy array
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    poly_on_mask = cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, poly_on_mask)                                                           # As we only required the pixels of lane and not the polygon, we can get it by comparing canny image with poly_on_masked using bitwise_and function
    return masked_image

# Function to display the detected lanes in form of lines over an image/frame

def display_lines(image, lines) :
    line_image = np.zeros_like(image)
    if lines is not None :
        for line in lines :
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)                                             # Display lanes i.e lines in a blue color over an image/frame 
    return line_image


# Working / calling of functions of detection and display of cars, peds and lanes over an video/live cam

while(video.isOpened()) :

    read_successful, frame = video.read()                                                                         # Stores a Boolean values and image

    if read_successful :
        car_peds_detection(frame)

        frame_np = np.copy(frame)
        canny_image = canny(frame_np) 
        cropped_image = region_of_interest(canny_image)

        # Here HoughLinesP is a method to detect max no of intersection of lines in a point in a bin over an image with 2 pixels, degree of 1, min no of intersections=100, minLineLenght and minLineGap to connect lines for optimization
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)   # Returns a list of array of values of pixels of the lines in the image
        line_image = display_lines(frame_np, lines)

        main_combine_image = cv2.addWeighted(frame_np, 0.8, line_image, 1, 1)                                     # 0.8 and 1 is used to add more intensity in pixels 
    else :
        break 

    cv2.imshow('Result', main_combine_image)

    # Since a for infinite while loop is used in the program, thus a key is assigned which can break the loop anytime when becomes True
    key = cv2.waitKey(1)
    if key == 81 or key == 113 :
        break


video.release()
cv2.destroyAllWindows()