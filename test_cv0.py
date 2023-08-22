import cv2
import numpy as np

# Create a black canvas (image) to draw on
height, width = 500, 500
canvas = np.zeros((height, width, 3), dtype='uint8')

# Define the rectangle's top-left and bottom-right coordinates
rect_top_left = (100, 100)
rect_bottom_right = (400, 300)

# Define the rectangle's color (BGR format)
rect_color = (0, 255, 0)  # Green color

# Draw the rectangle on the canvas
cv2.rectangle(canvas, rect_top_left, rect_bottom_right, rect_color, thickness=2)

# Display the canvas with the rectangle
cv2.imshow('Rectangle', canvas)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
