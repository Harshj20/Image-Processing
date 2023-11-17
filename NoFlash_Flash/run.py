import cv2
import numpy as np
import math

def solution(image_path_a, image_path_b):
#     ############################
#     ############################
#     ## image_path_a is path to the non-flash high ISO image
#     ## image_path_b is path to the flash low ISO image
#     ############################
    no_flash = cv2.imread(image_path_a)
    flash = cv2.imread(image_path_b)

    diameter = 25
    sigma_color = 3
    sigma_space = 9
    fused_image = joint_bilateral_filter(
            no_flash, flash, sigma_space, sigma_color
        )
    
    return fused_image *  detail_layer(flash)


def detail_layer(flash_img):

    flash_blurred = np.zeros_like(flash_img)
    for i in range(3):
        flash_blurred[:, :, i] = cv2.GaussianBlur(flash_img[:, :, i], (25, 25), 2)
    
    epsilon = 1e-10
    detail = flash_img / (flash_blurred + epsilon)    
    return detail


def joint_bilateral_filter(input_image, guidance_image, spatial_kernel, range_kernel):
    height, width, channels = input_image.shape
    original_image = np.pad(input_image, ((spatial_kernel, spatial_kernel), (spatial_kernel, spatial_kernel), (0, 0)), 'symmetric')
    second_image = np.pad(guidance_image, ((spatial_kernel, spatial_kernel), (spatial_kernel, spatial_kernel), (0, 0)), 'symmetric')
    
    gaussian_range = 1 / math.sqrt(2 * math.pi * (range_kernel ** 2))
    
    matrix = np.exp(-np.arange(256) * np.arange(256) * gaussian_range)

    output_image = np.zeros((height, width, channels), dtype=np.uint8)
    for x in range(spatial_kernel, spatial_kernel + height):
        for y in range(spatial_kernel, spatial_kernel + width):
            for i in range(channels):
                neighborhood = second_image[x - spatial_kernel : x + spatial_kernel + 1,
                                             y - spatial_kernel : y + spatial_kernel + 1, i]
                central = second_image[x, y, i]
                response = matrix[abs(neighborhood - central)]
                normalization = np.sum(response)
                output_image[x - spatial_kernel, y - spatial_kernel, i] = np.sum(
                    response * original_image[x - spatial_kernel : x + spatial_kernel + 1,
                                               y - spatial_kernel : y + spatial_kernel + 1, i]) / normalization

    return output_image