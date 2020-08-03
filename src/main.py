from argparse import ArgumentParser
from random import randint
from inference import *
import os
import sys
import time
import socket
import json
import cv2
import sys
import numpy as np
import logging

# reset && python3 main.py -d 'GPU' -fd ../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hp ../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -fl ../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -ge ../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i ../bin/demo.mp4 

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--facdec", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hp", "--hpest", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-fl", "--faclan", required=True, type=str,
                        help="Path to a facial landmarks xml file with a trained model.")
    parser.add_argument("-ge", "--gaze", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Path video file or CAM to use camera")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    parser.add_argument("--disp_out",default=True,type=bool,
                        help="display models output on frame")
    
    parser.add_argument("--mouse_con",default=True,type=bool,
                        help="move mouse based on gaze estimation output")
    
    parser.add_argument("--save_video",default=True,type=bool,
                        help="save video window")
    
    parser.add_argument("--show_video",default=False,type=bool,
                        help="show video window")
                        
    parser.add_argument("--input_type",default='video',type=str,
                        help="input type")
    

    return parser

def main():
    # Grab command line args
    args = build_argparser().parse_args()
    start = time.time()
    app = Inferencer(device=args.device,mouse_con=args.mouse_con, face_dec=args.facdec, fac_land=args.faclan, head_pose=args.hpest, gaze=args.gaze, show_video=args.show_video, save_video=args.save_video)
    app(input_type=args.input_type, input_file=args.input_file)
    print('program ends in {} mins'.format((time.time()-start)/60))
    
if __name__ == '__main__':
    main()