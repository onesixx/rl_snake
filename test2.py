import cv2

# Load an image from file
image = cv2.imread("imgs/snake.png")

# Display the image in a window
cv2.imshow("Image Window", image)

# Wait for a key press for a specified amount of time (in milliseconds)
# If a  key is pressed, the function returns the ASCII value of the key
# If no key is pressed within the given time, it returns -1
key_pressed = cv2.waitKey(0)

# Check if the key pressed is the 'Esc' key (ASCII value 27)
if key_pressed == 27:
    print("You pressed the 'Esc' key.")

# Close all OpenCV windows
cv2.destroyAllWindows()