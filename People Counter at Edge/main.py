"""People Counter."""

'''
To run the model-
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
'''
'''
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
'''
'''
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
'''
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
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
    return parser

def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    # Initialize the Inference Engine
    infer_network = Network()
    # Set Probability threshold for detections
    global prob_threshold
    prob_threshold = args.prob_threshold
    
    global width, height
    req_id = 0
    present_count = 0 #Number of people by the time have visited the frame
    total_count = 0 #Total people that had been in the frame throughout the video
    start_time = 0
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          req_id, 
                                          args.cpu_extension)[1]
    
    
    """
    Running People Counter on local machine
    """
    if args.input == 'CAM':
        input_stream = 0
    
    if args.input.endswith('.mp4'):
        input_stream = args.input
        assert os.path.isfile(args.input),"file unavailable"

    cap = cv2.VideoCapture(input_stream)
    if input_stream:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("Video parsing Error from the source")
   
    prob_threshold = args.prob_threshold
    width = cap.get(3)
    height = cap.get(4)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        """ 
        Preprocessing the obtained frame
        """
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        inf_start = time.time()
        infer_network.exec_net(req_id, image)

        if infer_network.wait(req_id) == 0:
            det_time = time.time() - inf_start
            result = infer_network.get_output(req_id)
         #   perf_count = infer_network.performance_counter(req_id)
            current_count = 0
            for obj in result[0][0]:
                if obj[2] > prob_threshold:
                    xmin = int(obj[3] * width)
                    ymin = int(obj[4] * height)
                    xmax = int(obj[5] * width)
                    ymax = int(obj[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    current_count = current_count + 1
            inf_time_message = "Inference time: {:.2f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            
            if current_count < present_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
                
            if current_count > present_count:
                start_time = time.time()
                total_count = total_count + current_count - present_count 
                client.publish("person", json.dumps({"total": total_count}))

            client.publish("person", json.dumps({"count": current_count}))
            present_count = current_count

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
   
        

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    
    
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()