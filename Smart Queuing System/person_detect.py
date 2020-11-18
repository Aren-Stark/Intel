
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    classes = {1: 'PERSON'}

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape 
        self.height,self.width = self.input_shape[2:]

    def load_model(self):
        self.plugin = IECore()
        self.net = self.plugin.load_network(self.model, self.device)
#         supported_layers = self.plugin.query_network(network=self.model, device_name=self.device)
#         unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
#         if len(unsupported_layers) != 0:
#             print("Unsupported layers found: {}".format(unsupported_layers))
#             print("Check whether extensions are available to add to IECore.")


    def predict(self, image):
        img = np.copy(image)
        h,w = image.shape[:2]
        preprocessed_image = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name: preprocessed_image})
        
        coords = self.preprocess_outputs(outputs,h,w)
        return coords,self.draw_outputs(coords,img)

    def draw_outputs(self, coords, image):
        for det in coords:
            x1, y1, x2, y2 = det[:4]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        return image

    def preprocess_outputs(self, outputs,h,w):
        def nms(dets, score_threshold=0.3, beta=3):
            def iou(bb_test, bb_gt):
                xx1 = np.maximum(bb_test[0], bb_gt[0])
                yy1 = np.maximum(bb_test[1], bb_gt[1])
                xx2 = np.minimum(bb_test[2], bb_gt[2])
                yy2 = np.minimum(bb_test[3], bb_gt[3])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w * h
                o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                          (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
                return (o)

            filtered_dets = []
            total_dets = sorted(dets.copy(), key=lambda x: x[4], reverse=True)
            while len(total_dets) > 0:
                for i in range(1, len(total_dets)):
                    IOU = iou(total_dets[i][:4], total_dets[0][:4])
                    total_dets[i][4] *= np.exp(-beta * IOU)  # (1-IOU)
                temp = []
                for i in total_dets:
                    if i[4] >= score_threshold:
                        temp.append(i)
                if len(temp) > 0:
                    filtered_dets.append(temp[0])
                total_dets = sorted(temp[1:].copy(), key=lambda x: x[4], reverse=True)
                del temp
            return filtered_dets
        dets = outputs[self.output_name][0][0]
        dets_fil = []
        for det in dets:
            if det[0] == -1:
                break
            else:
                if det[1] in list(self.classes.keys()) and float(det[2]) >= self.threshold:
                    x1, y1, x2, y2 = int(det[3] * w), int(det[4] * h), int(det[5] * w), int(det[6] * h)
                    if x1 < 0 : x1 = 0
                    if x2 <0 : x2 = 0
                    if y1<0: y1 = 0
                    if y2 <0: y2 = 0
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
        dets_fil = nms(dets_fil)
        return dets_fil

    def preprocess_input(self, image):
        image = np.copy(image)
        image = cv2.resize(image, (self.width, self.height))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, 3, self.height, self.width)
        return image


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)