import torch
import numpy as np
import cv2


def preprocess_frames(frame):
    """This function preprocesses Atari Pong Frames"""
    #Cropping
    frame[30:-12, 5:-4]

    #Grayscale
    frame = np.average(frame, axis=2)

    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)

    frame = np.array(frame, dtype=np.uint8)

    return torch.tensor(frame)


def stack_frames(state,frames):
    """This function stacks the four most recent frames"""

    frames.append(state)

    return frames
