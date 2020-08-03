from argparse import ArgumentParser
from random import randint
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import Head_Pose_Estimation
from gaze_estimation import Gaze_Estimation 
from mouse_controller import MouseController
from input_feeder import InputFeeder
import os
import sys
import time
import socket
import json
import cv2
import sys
import numpy as np
import logging

class Inferencer:
    def __init__(self, device='CPU',mouse_con=False, face_dec=None, fac_land=None, head_pose=None, gaze=None, show_video=False, save_video=False):
        '''
        all models should be put in here 
        '''
        if face_dec and fac_land and head_pose and gaze:
            self.face_dec, self.fac_land, self.head_pose, self.gaze = FaceDetectionModel(face_dec,device=device), FacialLandmarksDetection(fac_land,device=device), Head_Pose_Estimation(head_pose,device=device), Gaze_Estimation(gaze,device=device)
            self.face_dec.load_model()
            self.fac_land.load_model()
            self.head_pose.load_model()
            self.gaze.load_model()
        else:
            raise ValueError('Missing Arguments')
            
        if mouse_con:
            self.mouse_con = MouseController("low","fast") 
            
        self.show_video,self.save_video = show_video, save_video
    
    def __call__(self,input_type=None, input_file=None,):
        self.run(input_type=input_type, input_file=input_file)
        
    def run(self,input_type=None, input_file=None,): 
        if input_type and input_file:
            self.input_ = InputFeeder(input_type, input_file)
            self.input_.load_data()
            if self.save_video:
                out = cv2.VideoWriter('output.mp4', 0x00000021, 30, (int(self.input_.cap.get(3)),int(self.input_.cap.get(4))))
        try:
            fc_dec_inf_time = 0
            landmark_inf_time = 0
            pose_inf_time = 0
            gaze_inf_time = 0
            frame_counter = 0
            while True:
                # Read the next frame
                try:
                    frame = next(self.input_.next_batch())
                    frame_counter += 1
                except StopIteration:
                    break

                key_pressed = cv2.waitKey(60)

                # face detection
                start = time.time()
                out_frame,boxes = self.face_dec.predict(frame,display_output=True)
                fc_dec_inf_time += (time.time()-start)
                
                #for each box
                for box in boxes:
                    face = out_frame[box[1]:box[3],box[0]:box[2]]
                    
                    start = time.time()
                    out_frame,left_eye_point,right_eye_point = self.fac_land.predict(out_frame, face, box,display_output=True)
                    landmark_inf_time += (time.time()-start)
                    
                    start = time.time()
                    out_frame, headpose_angels = self.head_pose.predict(out_frame,face,box,display_output=True)
                    pose_inf_time += (time.time()-start)
                    
                    start = time.time()
                    out_frame, gazevector = self.gaze.predict(out_frame, face, box, left_eye_point, right_eye_point, headpose_angels, display_output=True)
                    gaze_inf_time += (time.time()-start)
                    
                    if self.show_video:
                        cv2.imshow('im', out_frame)
                        
                    if self.save_video:
                        out.write(out_frame)

                    if self.mouse_con:
                        self.mouse_con.move(gazevector[0],gazevector[1])
                    
                    time.sleep(1)

                    #consider only first detected face in the frame
                    break

                # Break if escape key pressed
                if key_pressed == 27:
                    break
                    
            if self.save_video:
                out.release()
            self.input_.close()
            cv2.destroyAllWindows()
            print('average inference time for face detection model is :- {:2f}ms'.format((fc_dec_inf_time/frame_counter)*1000))
            print('average inference time for facial landmark model is :- {:2f}ms'.format((landmark_inf_time/frame_counter)*1000))
            print('average inference time for head pose estimation model is :- {:2f}ms'.format((pose_inf_time/frame_counter)*1000))
            print('average inference time for gaze estimation model is :- {:2f}ms'.format((gaze_inf_time/frame_counter)*1000))
        except Exception as ex:
            logging.exception("Error in inference: " + str(ex))