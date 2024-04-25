import cv2
import numpy as np

def opening(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def dilatation(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def binarization(image):
    blue_channel = image[:, :, 0]
    median_filtered = cv2.medianBlur(blue_channel, 3)
    _, threshold = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    return threshold

def morphological(threshold):
    kernel_open = np.ones((2, 2), np.uint8)
    open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel_open, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations = 1)
    bg = cv2.dilate(closing, kernel, iterations = 1)
    dist_transform = cv2.distanceTransform(bg, cv2.DIST_L2, 0)
    ret, morphological_result = cv2.threshold(dist_transform, 0.02
                        * dist_transform.max(), 255, 0)
    morphological_result = morphological_result.astype(np.uint8)
    return morphological_result

def find_and_draw_connected_components(image, morphological_result):
    count = 0
    contours, _ = cv2.findContours(morphological_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 45:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
    return image, count
