| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.**6** |
| OpenVino Version: |  2020.**1.023** |
| Models Required: |face-detection-adas-binary-0001   <br /><br />landmarks-regression-retail-0009 <br /><br /> head-pose-estimation-adas-0001 <br /><br />gaze-estimation-adas-0002|

# Computer Pointer Controller

*TODO:* Write a short introduction to your project
Computer Pointer Controller is an IOT/AI program that aims to help control a computer's mouse pointer just by the movement of the eyes and the gaze of the eyes

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

The project heavily depends on openvino so, the first package to install is the openvino toolkit. also opencv should be installed on the system before this project can run.

This project also depends on three AI models, 

first you change directory in the model folder from the terminal if you are using linux/max or command line if you are using windows.

Next, run the following command

(1) Facedetection model
can be downloaded from openvino model zoo using the command --> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --precision FP32

(2) Facial Landmark Dectection
can be downloaded from openvino model zoo using the command --> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --precision FP32

(3) Head pose estimation
can be downloaded from openvino model zoo using the command --> python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --precision FP32

(4) gaze estimation
can be downloaded from openvino model zoo using the command -->  python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precision FP32


Once the models are downloaded the other required packages can be downloaded using the command --> pip install requirements.txt

## Demo
.
├── bin
│   └──  demo.mp4
├── models
├── README.md
├── requirements.txt
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── inference.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    └── mouse_controller.py

to run the demo use change directory to src from command line or terminal and type :-
python3 main.py -d 'CPU' -fd ../models/intel/face-detection-adas-binary-0001/FP32/face-detection-adas-binary-0001 -hp ../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -fl ../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -ge ../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 -i ../bin/demo.mp4
*TODO:* Explain how to run a basic demo of your model.

## Documentation
to see other arguments that can be passed to the model run this command in the src folder
python3 main.py -h

## Benchmarks

### Results for DEVICE = CPU
| Factor/Model       | Face Detection   | Landmarks Detetion        | Headpose Estimation | Gaze Estimation |
|--------------------|---------------|-----------|-------------|-----------|
|Load Time FP32-INT8 |  216ms        | NA        | NA          | NA        |
|Load Time FP32      |  NA           | 72ms      | 95ms        | 117ms     |
|Load Time FP16      |  NA           | 72ms      | 113ms       | 129ms     |  
||||||
|Inference Time FP32-INT8 | 52ms     | NA        | NA          | NA        |
|Inference Time FP32      | NA       | 1ms       | 3ms         | 3ms       |
|Inference Time FP16      | NA       | 1.7ms     | 3.9ms       | 3.6ms     |
||||||

### Results for DEVICE = IGPU
GPU system not available

## Results
* Load time for models with FP32 is less than FP16
* Inference time for models with FP32 is larger than FP16

