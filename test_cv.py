import cv2
import numpy as np
from IPython.display import display, Image

# Create a black img_canvas (image) to draw on
height, width = 500, 500
img_canvas = np.zeros((height, width, 3), dtype='uint8')

# Define the rectangle's top-left and bottom-right coordinates
rect_top_left     = ( 60, 150)
rect_bottom_right = ( 70, 160)
# Define the rectangle's color (BGR format)
rect_color = (0, 255, 0)  # Green color

# Draw the rectangle on the img_canvas
cv2.rectangle(img_canvas, 
              rect_top_left, rect_bottom_right, 
              rect_color, thickness=3)

# Convert the BGR image to RGB format
img_canvas_rgb = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)

# Save the image to a file
cv2.imwrite('imgs/temp.jpg', img_canvas_rgb)
display(Image(filename='imgs/temp.jpg'))
