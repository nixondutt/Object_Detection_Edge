import os 
import sys
import cv2
import yaml
from moviepy.editor import *
from utils.utils import *
from functools import wraps
import subprocess as sp
np.random.seed(0)

class Infer(Object):
    def __init__(self):
        self.opt = parse_opt()
        self.names = None
        self.clip = VideoFileClip(self.opt.video)
        self.interpreter = load_model(opt.weight)
        self.n_classes = 91
        self.colors = np.random.randint(0, 255, size=(self.n_classes, 3), 
                                        dtype="uint8")

    @timefunc
    def read_labels(self):
        with open(opt.label, errors = 'ignore') as f:
            self.names = yaml.safe_load(f)['names']

    def _set_input_tensor(self, image):
        """ sets the input tensor """
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self,index):
        """ Returns the output tensor at the given index """
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    @timefunc
    def detect_objects(self, image):
        """
        summary - Returns a list of detection results, each a dictionary of object info. 
        """
        self._set_input_tensor(image)
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter() - start_time
        print(f'interpreter forward function took {end_time:0.6f}s')
        # Get all output details
        boxes = self._get_output_tensor(0)
        classes = self._get_output_tensor(1)
        scores = self._get_output_tensor(2)
        count = int(self._get_output_tensor(3))
        # print(boxes)
        # print(classes)
        # print(scores)
        # print(count)
        results = []
        for i in range(count):
            if scores[i] >=self.opt.threshold and int(classes[i])==0:
                result = {
                    'bounding_box' : boxes[i],
                    'class_id' : classes[i],
                    'score' : scores[i]
                }
                results.append(result)
        return results

    @timefunc
    def draw(results,original_image):
        for obj in results:
            # convert the bouding box from relative cordinates to absolute cordinates
            ymin, xmin, ymax, xmax =  obj['bounding_box']
            xmin = int(xmin * original_image.shape[1])
            xmax = int(xmax * original_image.shape[1])
            ymin = int(ymin * original_image.shape[0])
            ymax = int(ymax * original_image.shape[0])

            # take class index
            idx = int(obj['class_id'])
            # skip the background
            if idx >= len(self.names):
                continue

            # draw the bounding box and label on the image
            color = [int(c) for c in self.colors[idx]]
            original_image = cv2.rectangle(original_image,(xmin,ymin),(xmax,ymax),color,2)
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            label = "{} : {:.2f}%".format(self.names[idx],obj['score'])
            # label = "{}%".format(names[idx])
            original_image = cv2.putText(original_image,label,(xmin,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
        
        return original_image 



    def infer_video(self,size):
        self.read_labels()
        count =0
        store_std_out = sys.stdout
        sys.stdout = open('log.txt','w')
        frames = self.clip.iter_frames()
        for original_image in frames:
            count += 1
            start_time = time.perf_counter()
            preprossed_image = cv2.resize(original_image.copy(), size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            input_data = np.expand_dims(preprossed_image, axis = 0)

            results = self.detect_objects(input_data)
            img = self.draw(results,original_image)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            inter_fps = int(1/(time.perf_counter() - start_time))

            
            cv2.rectangle(img, (50,6), (370, 60), (0,0,0), -1)
            cv2.putText(img, " FPS - {:.2f}".format(inter_fps),(40,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
            cv2.imwrite("data/write/img_"+str(count)+".png",img)
            print(f"frame number {count}, fps - {inter_fps}")

        sys.stdout.close()
        sys.stdout = store_std_out