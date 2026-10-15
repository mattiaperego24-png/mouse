import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import numpy as np
import os
import sys  

# function to use the .exe if u have it
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# SETUP SCREEN
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# SETTING CONNECTIONS
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (5,6), (6,7), (7,8),              # Index
    (9,10), (10,11), (11,12),         # Middle
    (13,14), (14,15), (15,16),        # Anular
    (17,18), (18,19), (19,20),        # Pinky
    (0,5), (5,9), (9,13), (13,17), (0,17) # Palm
]

# 3. INITIALIZATION DELL'AI
path_model = resource_path('hand_landmarker.task')      # search it on the internet to get more precise explanation on how is this ai working
base_options = python.BaseOptions(model_asset_path=path_model)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# STARTING THE WEBCAM
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
prev_index_y = 0
prev_basei_x, prev_basei_y = 0, 0

dragging = False
right_clicked = False
pinch_start_time = 0
chord_start = time.time()
chord_curr = 0
chord_sent = False
alphabet_time = time.time() 

# QWERTY 
show_keyboard = False
keyboard_cooldown = time.time()
keyboard_qwerty = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE'],
    ['BACK']
]

# SETTING WINDOW 
cv2.namedWindow("AI Mouse - Tasks API", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Mouse - Tasks API", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
# Optional: keep the window in the foreground 
# cv2.setWindowProperty("AI Mouse - Tasks API", cv2.WND_PROP_TOPMOST, 1)

# MAIN LOOP
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # SEARCHING FRAME
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # READING FRAME
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    detection_result = detector.detect(mp_image)

    # VARIABLES
    sxseen = 0 #flag
    dxseen = 0
    
    ix, iy, tx, ty, mx, my, ax, ay, px, py = [9999]*10
    ixl, iyl, txl, tyl, mxl, myl, axl, ayl, pxl, pyl = [9999]*10
    
    dist_fingersl = [999, 999, 999, 999, 999]
    dist_fingers = [999, 999, 999, 999, 999]


    if detection_result.hand_landmarks:
        for i in range(len(detection_result.hand_landmarks)):
            
            hand_landmarks = detection_result.hand_landmarks[i]
            hand_label = detection_result.handedness[i][0].category_name

            # DRAWING THE HAND CONNECTION
            for connection in HAND_CONNECTIONS:
                p1 = hand_landmarks[connection[0]]
                p2 = hand_landmarks[connection[1]]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (x1, y1), 4, (0, 0, 255), -1)

            # RIGHT HAND=MOUSE
            
            if (hand_label == "Left"):
                dxseen = 1
                
                # EXTRACTION OF KEY FINGERS POINT
                ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                tx, ty = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h)
                mx, my = int(hand_landmarks[12].x * w), int(hand_landmarks[12].y * h)
                ax, ay = int(hand_landmarks[16].x * w), int(hand_landmarks[16].y * h)
                px, py = int(hand_landmarks[20].x * w), int(hand_landmarks[20].y * h) 
                bix, biy = int(hand_landmarks[5].x * w), int(hand_landmarks[5].y * h) 

                # MAPPING (webcam-screen) AND SMOOTHING
                target_x = np.interp(bix, (150, w - 150), (0, screen_w))
                target_y = np.interp(biy, (150, h - 150), (0, screen_h))
                
                dist_mouse = math.hypot(target_x - prev_x, target_y - prev_y)
                if dist_mouse > 120:
                    dynamic_smoothing = 2.0  
                elif dist_mouse > 60:
                    dynamic_smoothing = 5.0 
                else:
                    dynamic_smoothing = 10.0 

                curr_x = prev_x + (target_x - prev_x) / dynamic_smoothing
                curr_y = prev_y + (target_y - prev_y) / dynamic_smoothing

                dist_pinch_left = math.hypot(tx - ix, ty - iy)   
                dist_pinch_right = math.hypot(tx - mx, ty - my)  
                
                for j in range(1,6):
                    if j < 5:
                        bx, by = int(hand_landmarks[(j+1)*4].x * w), int(hand_landmarks[(j+1)*4].y * h)
                    else:
                        bx, by = tx, ty
                    fx, fy = int(hand_landmarks[j*4].x * w), int(hand_landmarks[j*4].y * h)
                    dist_fingers[j-1] = math.hypot(fx - bx, fy - by)    

                # SCROLL AND MOVEMENT
                if (dist_fingers[0] < 75 and dist_fingers[1] < 75 and dist_fingers[2] < 75 and dist_fingers[3] < 75 and dist_fingers[4] < 75):
                    dy = prev_index_y - iy
                    cv2.circle(img, (px,py,), 15, (0, 0, 255), -1)
                    if abs(dy) > 2:
                        pyautogui.scroll(dy * 9)
                else:
                    if len(detection_result.hand_landmarks) == 1:
                        pyautogui.moveTo(curr_x, curr_y)
                    
                prev_index_y = iy

                # LEFT AND RIGHT CLICK
                if (dist_pinch_left < 40 and dist_fingers[4]>100 and not show_keyboard):
                    if not dragging:
                        if pinch_start_time == 0:
                            pinch_start_time = time.time()
                            pyautogui.click()
                            cv2.circle(img, (ix, iy), 15, (0, 255, 255), -1) 
                        elif time.time() - pinch_start_time > 0.8:
                            pyautogui.mouseDown()
                            dragging = True
                            cv2.circle(img, (ix, iy), 15, (0, 255, 125), -1) 
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    pinch_start_time = 0

                if (dist_pinch_right < 40 and dist_fingers[4]>100 and not show_keyboard):
                    if not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True
                    cv2.circle(img, (mx, my), 15, (255, 0, 0), -1) 
                else:
                    right_clicked = False
                    
                prev_x, prev_y = curr_x, curr_y

            # LEFT HAND = GESTURE (writing)

            elif (hand_label == "Right"):
                sxseen = 1
                
                ixl, iyl = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
                txl, tyl = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h)
                mxl, myl = int(hand_landmarks[12].x * w), int(hand_landmarks[12].y * h)
                axl, ayl = int(hand_landmarks[16].x * w), int(hand_landmarks[16].y * h)
                pxl, pyl = int(hand_landmarks[20].x * w), int(hand_landmarks[20].y * h) 

                for j in range(1,6):
                    if j <5:
                        bx, by = int(hand_landmarks[(j+1)*4].x * w), int(hand_landmarks[(j+1)*4].y * h)
                    else:
                        bx, by = txl, tyl
                    fx, fy = int(hand_landmarks[j*4].x * w), int(hand_landmarks[j*4].y * h) 
                    dist_fingersl[j-1] = math.hypot(fx - bx, fy - by) 

                # saving the number of fingers which are up in a list to add command based on that
                fingers_up_sx = []
                fingers_up_sx.append(1 if hand_landmarks[4].x > hand_landmarks[3].x else 0)
                for id in [8, 12, 16, 20]:
                    if hand_landmarks[id].y < hand_landmarks[id - 2].y:
                        fingers_up_sx.append(1)  
                    else:
                        fingers_up_sx.append(0)  
                
                if len(detection_result.hand_landmarks) == 1:   # this section (writing) has to work only when there is exactly 1 hand (left one)
                    
                    val = (fingers_up_sx[0] * 16) + (fingers_up_sx[1] * 8) + (fingers_up_sx[2] * 4) + (fingers_up_sx[3] * 2) + (fingers_up_sx[4] * 1)

                    alphabet = {
                        8: 'a',
                        4: 'e', 
                        1: 'i', 
                        16: 'o', 
                        6: ' ', 
                        24: 'u', 
                        20: 'c', 
                        17: 's', 
                        12: 't', 
                        9: 'r', 
                        5: 'n', 
                        31:'backspace', 
                        14: 'm', 
                        7: 'l', 
                        28: 'p', 
                        25: 'd', 
                        13: 'b', 
                        15: 'f', 
                        23: 'g', 
                        27: 'h', 
                        29: 'q', 
                        30: 'v', 
                        2: 'z',
                    }

                    if not show_keyboard:    # you can use left hand chord-writing if and only if you show to the webcam the left hand alone
                        if val != chord_curr:
                            chord_curr = val
                            chord_start = time.time()  
                            chord_sent = False       
                        
                        if time.time() - chord_start > 0.4 and not chord_sent: #time is necessary as an anti-spam and to avoid writing related problems
                            if val in alphabet:
                                letter = alphabet[val]
                                pyautogui.press(letter)
                                chord_sent = True 

        # TWO HANDS INTERACTION
        
        if (sxseen == 1 and dxseen == 1 and len(detection_result.hand_landmarks) == 2):
            
            # COPY (all left fingers down)
            if dist_fingersl[0] < 75 and dist_fingersl[1] < 75 and time.time() - alphabet_time > 0.8:   # also here there is a waiting time anti-spam
                pyautogui.hotkey('ctrl', 'c')
                cv2.putText(img, "Copiato!", (txl, tyl), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 208, 49), 2)
                alphabet_time = time.time()
                
            # PASTE (Peace sign with left hand, index and mmiddle finger up) 
            elif (fingers_up_sx == [0, 1, 1, 0, 0] and time.time() - alphabet_time > 0.8):
                pyautogui.hotkey('ctrl', 'v')
                alphabet_time = time.time()

            # CLOSE WINDOW (left middle finger up)
            elif (fingers_up_sx == [0, 0, 1, 0, 0] and time.time() - alphabet_time > 1.0):
                pyautogui.hotkey('alt', 'f4')
                alphabet_time = time.time()

            # CAPSLOCK (anular to thumb pinch)
            elif (math.hypot(txl-axl, tyl-ayl) < 40 and time.time() - alphabet_time > 0.8):
                pyautogui.hotkey('capslock')
                alphabet_time = time.time()
            
            # TOGGLE QWERTY KEYBOARD: touch left index with the right one
            if math.hypot(ix - ixl, iy - iyl) < 20 and time.time() - keyboard_cooldown > 0.8:
                show_keyboard = not show_keyboard  
                keyboard_cooldown = time.time()
                print(f"QWERTY: {'on' if show_keyboard else 'off'}")

        # DRAWING AND USING QWERTY KEYBOARD
        if show_keyboard:
            for row_idx, row in enumerate(keyboard_qwerty):
                for col_idx, letter in enumerate(row):
                        
                    x_key = 15 + (col_idx * 55)
                    y_key = 100 + (row_idx * 55) 
                    if letter == 'SPACE':
                        w_key = 150
                    elif letter == 'BACK':
                        w_key = 125
                    else:
                        w_key = 50
                        
                    cv2.rectangle(img, (x_key, y_key), (x_key + w_key, y_key + 50), (255, 0, 0), 2)
                    cv2.putText(img, letter, (x_key + 10, y_key + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                    # what if the right index pass over a key of the keyboard
                    if x_key < ix < (x_key + w_key) and y_key < iy < (y_key + 50):
                            
                        cv2.rectangle(img, (x_key, y_key), (x_key + w_key, y_key + 50), (255, 255, 255), -1)
                        cv2.putText(img, letter, (x_key + 10, y_key + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            
                        pinch_keyboard = math.hypot(tx - ix, ty - iy) # to select the letter you have to index to thumb pinch
                            
                        if pinch_keyboard < 40 and time.time() - keyboard_cooldown > 0.4:
                            cv2.rectangle(img, (x_key, y_key), (x_key + w_key, y_key + 50), (0, 0, 255), -1) 
                                
                            if letter == 'SPACE':
                                pyautogui.press('space')
                            elif letter == 'BACK':
                                pyautogui.press('backspace')
                            else:
                                pyautogui.press(letter.lower())
                                    
                            keyboard_cooldown = time.time()

    # 10. STOPPING THE PROGRAM AND SHOWING THE WEBCAM WINDOW
    cv2.imshow("AI Mouse - Tasks API", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):       # select the webcam window and press q to exit the program
        break

cap.release()
cv2.destroyAllWindows()