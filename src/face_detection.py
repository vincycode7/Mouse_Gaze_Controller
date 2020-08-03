'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2,time
class FaceDetectionModel:
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
        print('device is {}'.format(self.device))
        ### TODO: Add any necessary extensions ###
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(selff.extensions, self.device)
            
        print(self.model_structure, self.model_weights)
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
        
        print('load time for face detection model is :- {:2f}ms'.format((time.time()-start)*1000))

        #get useful information out of the network
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape
        
    def predict(self, image,display_output=True,threshold=0.5):
        '''
            TODO: You will need to complete this method.
            This method is meant for running predictions on the input image.
            '''
        self.display_output=display_output
        self.threshold = threshold
        self.image = image
        input_dict = self.preprocess_input(self.image)
        self.output = self.exc_net.infer(input_dict)
        return self.preprocess_output(self.image,self.output[self.output_name],display_output=self.display_output,threshold=self.threshold)

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

    def preprocess_output(self, image, outputs,display_output=True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        self.width,self.height = image.shape[1],image.shape[0]
        coords = outputs
        boxes = []
        for box in coords[0][0]:
            if box[2] >= threshold:
#                 boxes.append(box[3:])
                xmin = int(box[3]*self.width)
                ymin = int(box[4]*self.height)
                xmax = int(box[5]*self.width)
                ymax = int(box[6]*self.height)
                if display_output:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,100,225),2)
                    boxes.append([xmin, ymin,xmax, ymax])
        return image,boxes