# Real-Time American Sign Language Recognition





  
## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Creating Dataset](#dataset)
- [MediaPipe Holistics](#mediapipe-holistics)
- [LSTM model](#lstm-model)
- [Results](#results)
- [Future Scope](#future-scope)

## 1. Introduction

The aim of this project is to develop a real-time American Sign Language (ASL) detection system that utilizes the MediaPipe holistic approach in combination with a Long Short-Term Memory (LSTM) model. Capable of accurately recognizing and interpreting ASL gestures in real time, enabling effective communication between the deaf and hearing communities.

## 2. Dependencies

```bash
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import mediapipe as mp
```

### 3. Creating Dataset

Sign language for wordsare gestures in sequential data.
Recorded 30 videos each 30 frames long to illustrate the following words in ASL.

- 'hello'
- 'thanks'
- 'iloveyou'
- 'Book'
- 'drink'
- 'Computer'
-  'chair'
-  'candy'
-  'help'
-  'study'
-  'family'
-  'medicine'
-  'party'
-  'money'
-  'race'

Save data to path 

```bash
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou','Book', 'drink','Computer', 'chair','candy','help','study','family','medicine','party','money','race'])
# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
```

- 'Holistic' is a class that encapsulates the functionality for detecting facial landmarks, hand landmarks, and body pose landmarks in images or frames.
- 'min_detection_confidence' is a parameter that sets the minimum confidence threshold for the initial detection of landmarks. Landmarks with confidence scores below this threshold will not be detected.
- 'min_tracking_confidence' is a parameter that sets the minimum confidence threshold for tracking landmarks after the initial detection. Landmarks with confidence scores below this threshold will not be included in the tracking process.

  
```bash
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), #(x,y)
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(3000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
```
model is an object of mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

- In the above code snippet we craete 30 numpy array of keypoints detected from mediapipe holistioc by feeding contineous data one by one to the model.
- 
