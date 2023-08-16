import numpy as np  
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow 
"""
Input for Program: Image of a road with clearly labelled lanes
Expected Output: Lanes highlighted with white superimposed on the original image
Read the input images from the folder 'LD_test_imgs' and show the output images in the notebook.
You may check the following functions out:
1. cv2.inRange()
2. cv2.bitwise_or() and cv2.bitwise_and
3. cv2.fillPoly()
4. cv2.line()
5. cv2.addWeighted()
In addition, you may consider the structure of the program (functions that one may define) we discussed 
in the previous cell.

Fill in your code in place of 'pass'.
"""

def img2edge(img):
    gray_image = cv2.cvtColor((img), cv2.COLOR_BGR2GRAY)
    hsv_image  = cv2.cvtColor((img), cv2.COLOR_BGR2HSV)

    """
    Define ranges for 'yellow', pixels within this range will be picked
    """
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])

    """
    cv2.inRange(): Picks pixels from the image that are in the specified range
    """
    mask_y = cv2.inRange(hsv_image ,lower_yellow , upper_yellow)     
    mask_w = cv2.inRange(gray_image, 216, 255)
  
    """
    Compute Bitwise OR, combining both the white and yellow pixels
    """
    mask_yw = cv2.bitwise_or(mask_y, mask_w)

    """
    Compute Bitwise AND of mask_yw with gray_img, pixels that were yellow or 
    white will have the same intensity as the original grayscale image, the 
    other pixels will be removed.
    """
    mask_yw_image =  cv2.bitwise_and(mask_yw, gray_image)

    img_blur = cv2.GaussianBlur(mask_yw_image, (3, 3), 0)
    img_canny = cv2.Canny(img_blur, 70, 200)
   # cv2_imshow(img_canny)

    return img_canny
    
def roi_select(img, canny):
    """
    Define the vertices of the region of interest
    """
    lower_left = 1
    lower_right = 1
    top_left = 100
    top_right = 100
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    
    mask = np.zeros_like(canny)
                         #creates a numpy array of the same dimensions as img
    #mask = np.zeros([240,320],dtype=np.uint8)
    fill_color = 255                                #parameter for cv2.fillPoly function
    cv2.fillPoly(mask, vertices, fill_color)        #pixels within 'vertices' in 'mask' will be made WHITE while all other pixels will be BLACK
    #cv2_imshow( mask1)
    return cv2.bitwise_and(canny, mask)

def draw_lines(canny_roi, rho_acc, theta_acc, thresh, minLL, maxLG):
    """
    Inputs - canny_roi 
    Parameters of HoughLinesP() are passed in as parameters to draw_lines()
    Output - line_img (image of lines against a black background)

    Perform Probabilistic Hough Transform on it, draw lines on a blank image using the values 
    returned by HoughLinesP() and the openCV function cv2.line(). 
    """ 
    #img1= canny_roi
    lines=cv2.HoughLinesP(canny_roi, rho_acc, theta_acc, thresh, minLL, maxLG)
    for x1,y1,x2,y2 in lines[0]:
         line_img=cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),5)
    return line_img

def add_weighted(img, line_img):
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

"""
The below code is for a single image, perform the same procedure for all the images in the
directory 'LD_test_imgs' and display all of them in  output of the cell. 
"""
img = cv2.imread("/content/CVI-Project-apps/CV_PS_imgs/LD_test_imgs/test_img01.jpeg",1)
#Read the input image from the directory.
edge_img = img2edge(img)
roi_img = roi_select(img, edge_img)
hough_img = draw_lines(roi_img, 2, np.pi/180, 50, 50, 100)      #Change the parameters thresh, minLL, maxLG to get more accurate lines
lane_img = add_weighted(img, hough_img)
cv2_imshow(lane_img)
