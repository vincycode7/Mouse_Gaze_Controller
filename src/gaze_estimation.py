'''
    This is a sample class for a model. You may choose to use it as-is or make any changes to it.
    This has been provided just to give you an idea of how to structure your model class.
    '''
from openvino.inference_engine import IENetwork, IECore
import cv2,time
class Gaze_Estimation:
    '''
        Class for the Face Detection Model.
        '''
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.5):
        '''
            TODO: Use this to set your instance variables.
            '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extensions = extensions
        self.threshold = threshold
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
        
        print('load time for gaze estimation model is :- {:2f}ms'.format((time.time()-start)*1000))

        #get useful information out of the network
        self.input_name=list(self.model.inputs)
        self.input_shape1=self.model.inputs[self.input_name[0]].shape
        self.input_shape2=self.model.inputs[self.input_name[1]].shape
        self.input_shape3=self.model.inputs[self.input_name[2]].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def predict(self, image, face, face_cords, left_eye_point, right_eye_point, headpose_angels, display_output=True):
        '''
            TODO: You will need to complete this method.
            This method is meant for running predictions on the input image.
            '''
        self.display_output=display_output
        self.image = image
        out_frame, left_eye, right_eye = self.preprocess_input(image, face, left_eye_point, right_eye_point, display_output=True)
        input_dict = {self.input_name[0] : headpose_angels, self.input_name[1] : left_eye, self.input_name[2] : right_eye}
        self.output = self.exc_net.infer(input_dict)
        return self.preprocess_output(out_frame, self.output[self.output_name],face_cords, left_eye_point, right_eye_point,display_output=True)

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image, face, left_eye_point, right_eye_point, display_output=True):
        '''
            Before feeding the data into the model for inference,
            you might have to preprocess it. This function is where you can do that.
           Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width
            with the name left_eye_image and the shape [1x3x60x60].
            Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width
            with the name right_eye_image and the shape [1x3x60x60].
            Blob in the format [BxC] where:
            B - batch size
            C - number of channels
            with the name head_pose_angles and the shape [1x3].
            '''
        
        #assign input shape
        lefteye_input_shape, righteye_input_shape =  [1,3,60,60], [1,3,60,60]
        
        # parse input shape
        image,pleftframe = self.crop_eye(image,lefteye_input_shape, left_eye_point,face,display_output)
        image,prightframe = self.crop_eye(image,righteye_input_shape, right_eye_point,face,display_output)

        return image, pleftframe, prightframe

    def crop_eye(self,image,eye_input_shape, eye_point, face,display_output=True):
                #crop eye
                xcenter,ycenter, width, height,facewidthedge, faceheightedge = eye_point[0], eye_point[1], eye_input_shape[3], eye_input_shape[2],face.shape[1], face.shape[0]

                # check for edges to not crop
                ymin = int(ycenter - height//2) if  int(ycenter - height//2) >=0 else 0 
                ymax = int(ycenter + height//2) if  int(ycenter + height//2) <=faceheightedge else faceheightedge

                xmin = int(xcenter - width//2) if  int(xcenter - width//2) >=0 else 0 
                xmax = int(xcenter + width//2) if  int(xcenter + width//2) <=facewidthedge else facewidthedge

                eye_image = face[ymin: ymax, xmin:xmax]

                #display eye to frame
                if display_output:
                    image[150:150+eye_image.shape[0],20:20+eye_image.shape[1]] = eye_image
                    # left eye [1x3x60x60]
                    pframe = cv2.resize(eye_image, (eye_input_shape[3], eye_input_shape[2]))
                    pframe = pframe.transpose((2,0,1))
                    pframe = pframe.reshape(1, *pframe.shape)
                    return image,pframe
                
    def preprocess_output(self, image, outputs, face_cords, left_eye_point, right_eye_point,display_output=True):
        '''
            Before feeding the output of this model to the next model,
            you might have to preprocess the output. This function is where you can do that.
            The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
            Output layer name in Inference Engine format:
            gaze_vector
            '''
        x,y,z = outputs[0][0], outputs[0][1], outputs[0][2]
        
        #Draw output
        if display_output:
            cv2.putText(image,"x:"+str('{:.1f}'.format(x*100))+",y:"+str('{:.1f}'.format(y*100))+",z:"+str('{:.1f}'.format(z)) , (20, 100), 0,0.6, (0,0,255), 1)

            #left eye
            xmin,ymin,_,_ = face_cords
            xcenter,ycenter = left_eye_point[0], left_eye_point[1]
            left_eye_centerx,left_eye_centery = int(xmin + xcenter), int(ymin + ycenter)
            
            #right eye
            xcenter,ycenter = right_eye_point[0],right_eye_point[1]
            right_eye_centerx, right_eye_centery = int(xmin + xcenter), int(ymin + ycenter)

            cv2.arrowedLine(image, (left_eye_centerx,left_eye_centery), (left_eye_centerx + int(x*100),left_eye_centery + int(-y*100)), (255, 100, 100), 5)
            cv2.arrowedLine(image, (right_eye_centerx,right_eye_centery), (right_eye_centerx + int(x*100),right_eye_centery + int(-y*100)), (255,100, 100), 5)

        return image, [x, y, z]