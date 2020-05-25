import os
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp


import cv2
from darkflow.net.build import TFNet
options = {"model": "cfg/yolo.cfg", 
           "load": "weights/yolo.weights", 
           "threshold": 0.1}

tfnet = TFNet(options)


def boxing(original_img, predictions):
  newImage = np.copy(original_img)

  for result in predictions:
      top_x = result['topleft']['x']
      top_y = result['topleft']['y']

      btm_x = result['bottomright']['x']
      btm_y = result['bottomright']['y']

      confidence = result['confidence']
      label = result['label'] + " " + str(round(confidence, 3))

      if confidence > 0.3:
          newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
          newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
          
return newImage



original_img = cv2.imread("./sample_img/sample_multiple_objects.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(original_img)

pp.pprint(results)



original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

cv2.imshow('image',original_img)

while(True):
    k = cv2.waitKey(33)
    if k == -1:  # if no key was pressed, -1 is returned
        continue
    else:
        break
cv2.destroyWindow('img')

# fig, ax = plt.subplots(figsize=(10, 10))



# ax.imshow(original_img)
# fig, ax = plt.subplots(figsize=(20, 10))
# newImage = boxing(original_img, results)
# cv2.imshow('image',newImage)


cap = cv2.VideoCapture('./sample_video/test_video.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# fourcc = cv2.c(*'DIVX')
# out = cv2.VideoWriter('./sample_video/output.avi',fourcc, 20.0, (int(width), int(height)))
# out = cv2.VideoWriter('./sample_video/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./sample_video/output.avi',fourcc, 20.0, (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        
        frame = np.asarray(frame)
        
        results = tfnet.return_predict(frame)
        
        
        new_frame = boxing(frame, results)
        cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
        print(new_frame.shape)
       
        # Display the resulting frame
        
        out.write(new_frame)
        
        #cv2.imshow('frame',new_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()



