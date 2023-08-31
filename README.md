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

Recorded 30 videos each 30 seconds long to illustrate the following words in ASL.

- Book
- Candy
- Chair
- Computer
- Drink
- Family
- Hello
- Help
- I Love you
- Medicine
- Money
- Party
- Race
- Study
- Thanks


