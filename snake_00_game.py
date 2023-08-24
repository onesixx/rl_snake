# https://theailearner.com/2019/03/10/creating-a-snake-game-using-opencv-python/
# https://github.com/TheAILearner/Snake-Game-using-OpenCV-Python/blob/master/snake_game_using_opencv.ipynb

import numpy as np
import cv2
import random
import time

# 1. die 
def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1  # true
    else:
        return 0 
# 2. die 
def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0
    
# 3-1. Score increases and apple is moved to new position
def collision_with_apple(apple_position, score):
    score += 1
    apple_position = [random.randrange(1,50)*10,
                        random.randrange(1,50)*10]
    return apple_position, score

### Game Window :: Display game objects
img = np.zeros((500,500, 3), dtype='uint8')
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

### Apple and Snake - Initial positions :: Display game objects 
apple_position = [random.randrange(1,50)*10, random.randrange(1,50)*10]
snake_position = [[250,250],[240,250],[230,250]]

score = 0
prev_button_direction = 1
button_direction = 1
snake_head = [250,250]

while True:
    cv2.imshow('a',img)
    cv2.waitKey(1)
    img = np.zeros((500,500,3), dtype='uint8')

    # Display Apple
    cv2.rectangle(img,(apple_position[0], apple_position[1]), (apple_position[0]+10,apple_position[1]+10), (0,0,255), 3)
    # Display Snake
    for position in snake_position:
        cv2.rectangle(img,(position[0],position[1]), (position[0]+10,position[1]+10),(0,255,0),3)

    # Takes step after fixed time
    t_end = time.time() + 0.2      # from 0.05
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(125)   # from 1
        else:
            continue

    # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
    # a-Left, d-Right, w-Up, s-Down
    if   k == ord('a') and prev_button_direction != 1:
        button_direction = 0
    elif k == ord('d') and prev_button_direction != 0:
        button_direction = 1
    elif k == ord('w') and prev_button_direction != 2:
        button_direction = 3
    elif k == ord('s') and prev_button_direction != 3:
        button_direction = 2
    elif k == ord('q'):
        break
    else:
        button_direction = button_direction
        
    prev_button_direction = button_direction

    # Change the head position based on the button direction
    if   button_direction == 1:
        snake_head[0] += 10
    elif button_direction == 0:
        snake_head[0] -= 10
    elif button_direction == 2:
        snake_head[1] += 10
    elif button_direction == 3:
        snake_head[1] -= 10

    # 3-2. Increase Snake length on eating apple
    if snake_head == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0,list(snake_head))
    else:
        snake_position.insert(0,list(snake_head))
        snake_position.pop()

    ### Displaying the final Score  
    ### On collision kill the snake and print the score
    if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((500,500,3), dtype='uint8')
        cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('a',img)
        cv2.waitKey(0)
        #cv2.imwrite('D:/downloads/ii.jpg',img)
        break

cv2.destroyAllWindows()