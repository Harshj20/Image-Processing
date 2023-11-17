import cv2
import numpy as np

# Usage
def solution(image_path):
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 120, 100])
    upper_red = np.array([50, 255, 255])
    
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    
    red_only_image = cv2.bitwise_and(image, image, mask=mask_red)
    
    gray_image = cv2.cvtColor(red_only_image, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    closing_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # smoothed_mask = cv2.GaussianBlur(closing_mask, (3, 3), 0)
    smoothed_mask = cv2.bilateralFilter(closing_mask, d=9, sigmaColor=75, sigmaSpace=25)
    
    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 300  
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    final_mask = np.zeros_like(gray_image)
    
    cv2.drawContours(final_mask, large_contours, -1, (255), thickness=cv2.FILLED)
    
    # final_mask = cv2.bitwise_not(final_mask)
    final_mask = cv2.merge([final_mask, final_mask, final_mask])
    return final_mask