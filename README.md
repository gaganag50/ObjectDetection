# ObjectDetection
## Detect objects in a video and displays them using Mpimg in the notebook
### Objects in image:

[<img src="sample_img/sample_multiple_objects.jpg" width=300>]()

[<img src="sample_img/savedImage.jpg" width=300>]()

### Objects in video:

![sample_video](https://github.com/gaganag50/ObjectDetection/blob/master/sample_video/sample.gif)
![sample_video_output](https://github.com/gaganag50/ObjectDetection/blob/master/sample_video/sample_output.gif)

### How to use:
run the yolo.py file and give required paths

python yolo.py -m cfg/yolo.cfg -w weights/yolo.weights -l cfg/coco.names -v test_video.mp4  -o sample_video/output.webm
