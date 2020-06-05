import os
import numpy as np
from darkflow.net.build import TFNet
import cv2
import argparse

from moviepy.editor import *

def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    for box in predictions:
        
        x1,y1,x2,y2 = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
        conf = box['confidence']
        
        label = box['label']
       
        if conf < predictThresh:
            continue
        color = [int(c) for c in colors[label]]
        
        cv2.rectangle(newImage,(x1,y1),(x2,y2),color,6)
        labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_COMPLEX,0.5,2)
        
        _x1 = x1
        _y1 = y1
        _x2 = _x1+labelSize[0][0]
        _y2 = y1-int(labelSize[0][1])
        
        cv2.rectangle(newImage,(_x1,_y1),(_x2,_y2),color,cv2.FILLED)
        cv2.putText(newImage,label,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    return newImage

def get_output(frame):
    frame = np.asarray(frame)
    results = tfnet.return_predict(frame)
    new_frame = boxing(frame, results)
    return new_frame

def get_color_per_label(labelsPath):        
    labels = None
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')


    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),dtype="uint8")


    colors = {}
    for label, color in zip(labels, COLORS):
        colors[label] = color

    return colors



    
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model", required=True, help="model path")
ap.add_argument("-w", "--weights", required=True, help="weights path")
ap.add_argument("-l", "--labels", required=True, help="labels path")

ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")

ap.add_argument("-v", "--video",  required=True,  help="path to video")
ap.add_argument("-o", "--output",  required=True,  help="path to output the video")


args = vars(ap.parse_args())
options = {"model": args["model"], 
           "load": args["weights"], 
           "threshold": args["confidence"]}
tfnet = TFNet(options)
colors = get_color_per_label(args["labels"])

my_clip = VideoFileClip(args["video"])

predictThresh = args["threshold"]
modifiedClip = my_clip.fl_image(get_output)

modifiedClip.write_videofile(args["output"])