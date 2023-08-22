from collections import deque

# Create a deque with a maximum length of 6
my_deque = deque(maxlen=6)
my_deque
deque(maxlen = 30)

# Add elements to the deque
my_deque.append(1)
my_deque.append(2)
my_deque.append(3)
# Current state of the deque: deque([1, 2, 3], maxlen=6)

# Add more elements, exceeding the maximum length
my_deque.append(4)
my_deque.append(5)
my_deque.append(6)
my_deque.append(7)
# Current state of the deque: deque([3, 4, 5, 6, 7], maxlen=6)

for i in range(30):
    self.prev_actions.append(-1) # to create history

list(prev_actions)
prev_actions = deque(maxlen = 30)  # however long we aspire the snake to be
for i in range(30):
    prev_actions.append(-1) # to create history

import gymnasium
# Create the LunarLander environment
env = gymnasium.make('LunarLander-v2', render_mode='human')
# Reset the environment
env.reset()
# Render the environment (a separate window should appear)
env.render()
# Optionally, interact with the environment, take actions, etc.
# Close the environment once done
env.close()

cv2.imshow('a', img)
np.zeros( (2, 5,3),dtype='uint8') 


from IPython.display import display, Image
import numpy as np
import cv2
img = np.zeros((500, 500,3),dtype='uint8')
# Convert image from BGR to RGB (OpenCV uses BGR by default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

display(Image(data=img_rgb))

from IPython.display import display, Image
import numpy as np
import cv2

# Create a gradient pattern
height, width = 500, 500
img = np.zeros((height, width, 3), dtype='uint8')

for y in range(height):
    for x in range(width):
        img[y, x] = [x, y, 255]

# Convert image from BGR to RGB (OpenCV uses BGR by default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the gradient image
display(Image(data=img_rgb))


# Save the image to a file
cv2.imwrite('img/temp.jpg', canvas_rgb)
display(Image(filename='img/temp.jpg'))

img = cv2.rectangle(img, 
        (255, 255),
        (255+10, 255+10),
        (0,255,0), 3)