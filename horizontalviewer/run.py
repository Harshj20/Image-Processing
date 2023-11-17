import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################
    canvas = cv2.imread(image_path)
    padding = 50
    (h, w) = canvas.shape[:2]

    new_h = h + 2 * padding
    new_w = w + 2 * padding
    image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255

    x_offset = padding
    y_offset = padding

    image[y_offset:y_offset + h, x_offset:x_offset + w] = canvas

    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.bitwise_not(gray_scale)

    _, inverted = cv2.threshold(gray_scale, 0, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(inverted, threshold1=50, threshold2=100)
    detected_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=10)
    if detected_lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for line in detected_lines for x1, y1, x2, y2 in [line[0]]]

        unique_angles, counts = np.unique(angles, return_counts=True)
        angle = unique_angles[np.argmax(counts)]

        line = detected_lines[0][0]
        mid_x = (line[0] + line[2]) // 2
        mid_y = (line[1] + line[3]) // 2
        roi_size = 5
        roi_x1 = max(mid_x - roi_size // 2, 0)
        roi_x2 = min(mid_x + roi_size // 2, gray_scale.shape[1])
        roi_y1 = max(mid_y - roi_size // 2, 0)
        roi_y2 = min(mid_y, gray_scale.shape[0]) 

        roi_upward = gray_scale[roi_y1:roi_y2, roi_x1:roi_x2]

        nonZero_pixels = cv2.countNonZero(roi_upward)
        if nonZero_pixels > 0:
            angle = angle - 180

        image_height, image_width = image.shape[:2]
        center_point = (image_width // 2, image_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (image_width, image_height),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
