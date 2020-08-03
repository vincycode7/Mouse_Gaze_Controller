'''
    This is a sample class for a model. You may choose to use it as-is or make any changes to it.
    This has been provided just to give you an idea of how to structure your model class.
    '''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2, math,time
class Head_Pose_Estimation:
    '''
        Class for the Face Detection Model.
        '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
            TODO: Use this to set your instance variables.
            '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions = extensions
        self.width=None
        self.height=None
        
    def load_model(self):
        '''
            TODO: You will need to complete this method.
            This method is for loading the model to the device specified by the user.
            If your model requires any Plugins, this is where you can load them.
            '''
        start = time.time()
        #Initializing the IECore
        self.plugin = IECore()
        
        ### TODO: Add any necessary extensions ###
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(selff.extensions, self.device)
            
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
        
        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check if any extensions is available for the device.")
            exit(1)
            
        
        # Load IENetwork into IECore
        self.exc_net = self.plugin.load_network(self.model, self.device,num_requests=1)
        
        print('load time for head pose estimation is :- {:2f}ms'.format((time.time()-start)*1000))

        #get useful information out of the network
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def predict(self, image,face,face_cords=None,display_output=True):
        '''
            TODO: You will need to complete this method.
            This method is meant for running predictions on the input image.
            '''
        self.display_output=display_output
        self.image = image
        self.face = face
        self.face_cords = face_cords
        input_dict = self.preprocess_input(self.face)
        self.output = self.exc_net.infer(input_dict)
        return self.preprocess_output(self.image,self.output,self.face,self.face_cords,display_output=self.display_output)

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
            Before feeding the data into the model for inference,
            you might have to preprocess it. This function is where you can do that.
            '''
        # [1x3x384x672]
        n, c, h, w = self.input_shape
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
        resized_frame = image.transpose((2, 0, 1)).reshape((n, c, h, w))
        return {self.input_name:resized_frame}

    def preprocess_output(self,image,outputs,face, face_cords, display_output=True):
        '''
            Before feeding the output of this model to the next model,
            you might have to preprocess the output. This function is where you can do that.
            Output layer names in Inference Engine format:
            name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
            name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
            name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
            Each output contains one float value  (yaw, pitсh, roll).
            '''
        
        angle_y,angle_p,angle_r = outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]   # Axis of rotation: z, Axis of rotation: y, Axis of rotation: x
         
        #Draw output
        if display_output:
            cv2.putText(image,"y:{:.1f}".format(angle_y), (20,20), 0, 0.6, (255,255,0))
            cv2.putText(image,"p:{:.1f}".format(angle_p), (20,40), 0, 0.6, (255,255,0))
            cv2.putText(image,"r:{:.1f}".format(angle_r), (20,60), 0, 0.6, (255,255,0))
            
            xmin, ymin,_ , _ = face_cords
            face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
            self.draw_axes(image, face_center, angle_y, angle_p, angle_r)
        
        return image, [angle_y, angle_p, angle_r]
     
     # code source: https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll):
        focal_length = 950.0
        scale = 50

        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame
    
    # code source: https://knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix