# 🖱️ Webcam Hand Mouse Controller

Transform your hands into a fully functional computer mouse! 

This repository contains a Python script that uses Google's MediaPipe (`hand_landmarker.task`) to track your hand movements via your webcam and translate them into system commands. 

## ✨ Features
Control your PC entirely hands-free! By moving your hands in front of the webcam, you can:
- **Move the cursor**
- **Click & Scroll**
- **Type / Write**
- **Quick Shortcuts:** Copy, paste, and close windows

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. You will need to install the required libraries (like `mediapipe`, `cv2`,`numpy` and `pyautogui`). You can usually install them via terminal:
`pip install opencv-python mediapipe pyautogui numpy`

### Usage
1. Clone this repository to your local machine.
2. Ensure the `hand_landmarker.task` file is in the same directory as the script.
3. Run the Python file:
   `python nome_del_tuo_file.py` *(Change this to your actual file name)*
4. Keep your hands in the webcam's view and start controlling your PC!

## 🧠 Explore the Code
Dive into the code! I've included comments to explain the logic behind the gesture recognition and show exactly how it works. 

Feel free to explore, tweak the parameters, or fork the project to add your own custom gestures and features. Enjoy!
