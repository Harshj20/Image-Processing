import cv2
import numpy as np

def preprocess_image(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    _, thresholded_image = cv2.threshold(gray_image, 1, 250, cv2.THRESH_BINARY)

    return thresholded_image

# Usage
def solution(image_path):
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    input_image = cv2.imread(image_path)
    height, width, _ = input_image.shape
    
    processed_image = preprocess_image(input_image)

    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = contours[0]
        epsilon = 0.04 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, epsilon, True)
        points = [(0, 0), (0, height), (width, height), (width, 0)]

        if len(corners) == 4:
            sorted_corners = [None] * 4
            for i in range(4):
                sample = [np.sqrt((corner[0][0] - points[i][0]) ** 2 + (corner[0][1] - points[i][1]) ** 2) for corner in corners]
                sorted_corners[i] = corners[np.argmin(sample)]

            target_width, target_height = 600, 600

            target_corners = np.array([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]], dtype=np.float32)

            perspective_matrix = cv2.getPerspectiveTransform(np.float32(sorted_corners), target_corners)

            transformed_image = cv2.warpPerspective(input_image, perspective_matrix, (target_width, target_height))

            return transformed_image
        else:
            print("Could not find 4 corners in the processed image.")
    else:
        print("No contours found in the processed image.")

