'''
    This is a sample class for a model. You may choose to use it as-is or make any changes to it.
    This has been provided just to give you an idea of how to structure your model class.
    '''
from openvino.inference_engine import IENetwork, IECore
import cv2,time
class FacialLandmarksDetection:
    '''
            Class forace Detection Model.
        '''
    def __init__(self, model_name, device='CPU', extensions=None,threshold=0.5):
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
        
        print('load time for facial land mark model is :- {:2f}ms'.format((time.time()-start)*1000))

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
        input_dict = self.preprocess_input(self.face)
        self.output = self.exc_net.infer(input_dict)
        self.face_cords = face_cords
        return self.preprocess_output(self.image,self.output[self.output_name],self.face_cords,display_output=self.display_output)

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

    def preprocess_output(self,image,outputs,face_cords, display_output=True):
        '''
            Before feeding the output of this model to the next model,
            you might have to preprocess the output. This function is where you can do that.
            '''
    
        landmarks = outputs.reshape(1, 10)[0]
        width,height = face_cords[2]-face_cords[0],face_cords[3]-face_cords[1] #ymax-ymin

        # Draw the face_boxes 
        if display_output:
            for each_lmrks in range(2):
                x = int(landmarks[each_lmrks*2] * width)
                y = int(landmarks[each_lmrks*3] * height)  
                cv2.circle(image, (face_cords[0]+x, face_cords[1]+y), 30, (0,255,255), 2)
        
        left_eyep =[landmarks[0] * width,landmarks[1] * height]
        right_eyep = [landmarks[2] * width,landmarks[3] * height]
        return image, left_eyep, right_eyep