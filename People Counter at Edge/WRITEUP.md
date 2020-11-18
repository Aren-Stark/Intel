## Project Write-Up
##1.Install OpenVino
Use Workspace Instead


##2. Install Nodejs and its dependencies
Use Workspace Instead


##3.Install web server npm:
1.For MQTT/Mosca server:
cd webservice/server
npm install

1.For Web server:
cd webservice/ui
npm install


##4.Have four terminal to run your project
Step 1 - Start the Mosca server
cd webservice/server/node-server
node ./server.js

Step 2 - Start the GUI
cd webservice/ui
npm run dev

Step 3 - FFmpeg Server
sudo ffserver -f ./ffmpeg/server.conf


##5.Setup the environment
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5





## Model Research
##6.Downloading and Processing Intermediate Representation for TensorFlow Models
/*Though I have worked with MXNET & ONNX-> fOR PyTorch but here I have listed my three best conclusions*/
pip3 install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements_tf.txt


##7.Configuring models

1.a>Downloading ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz -P /home/workspace/model
cd /home/workspace/model
mkdir ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03_IR
tar -xvf /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz
cd /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels -o /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03_IR
##b>Testing this model
cd /home/workspace
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
python main.py -i resources/test.mp4 -m /home/workspace/model/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


2.a>Downloading ssdlite_mobilenet_v2_coco_2018_05_09
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz -P /home/workspace/model
cd /home/workspace/model
mkdir ssdlite_mobilenet_v2_coco_2018_05_09_IR
tar -xvf /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
cd /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels -o /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09_IR
##b>Testing this model
cd /home/workspace
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
python main.py -i resources/test.mp4 -m /home/workspace/model/ssdlite_mobilenet_v2_coco_2018_05_09_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


3.a>Downloading ssd_mobilenet_v2_coco
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -P /home/workspace/model
cd /home/workspace/model
mkdir ssd_mobilenet_v2_coco_2018_03_29_IR
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_use_custom_operations_config  /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels -o /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29_IR
##b>Testing this model
cd /home/workspace
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
python main.py -i resources/test.mp4 -m /home/workspace/model/ssd_mobilenet_v2_coco_2018_03_29_IR/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


4.a>Downloading person-detection-retail-0013
python3  /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precison FP16 -o /home/workspace/model
##b>Running App
python main.py -i resources/test.mp4 -m /home/workspace/model/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/model/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm





## Explaining Custom Layers
1.The process behind converting custom layers involves three stages while the entire conversion is a five stage process:
/* Here I'm detailing about tensorflow*/
Custom Layers are basically off developed part for a model-> Since uncompatible by default they need to be provide support by developer-> Also termed as Unsupported layer-> By default MO cannot provide IR for these unsupported layers, So for tensorflow we'll do:
a.Generate the Extension Template Files Using the Model Extension Generator:
python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir=$<path>/<custom_layer_output_folder>
/*Two folder will appear one for model optimisation extension and second for inference engine extension*/


b.Generate IR Files including extension layers using MO:
1>Edit the extractor extension template file 
2>Edit the operation extension template file 
3>Generate the Model IR Files


c.Execute the Model with the Custom Layer:
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i $<path_to_created_TensorFlow_model_extension_layer.>/picture/image.bmp -m $<path_to_folder&unsupported_layer>/<custom_layer_output_extension_folder>/model.ckpt.xml -l $<path_to_folder&unsupported_layer>/<custom_layer_output_folder>/user_ie_extensions/cpu/build/libcosh_cpu_extension.so -d CPU

2.Some of the potential reasons for handling custom layers are...
Processed Custom Layers after Inferencing able to built with model are considered to be Stubs-> Stubs are basically ideal to foresee how entire process will work with unpredicted condition-> at present being at such an aerly stage no model it fully polished so giving a way to use unsupported layers as extensions and improve the entire work.





## Comparing Model Performance
My method(s) to compare models before and after conversion to Intermediate Representations
were FPS, Inference Timing, Loseless-Accuracy(/*Although a little bit loosy accuray but I have tried tp figure out the model with whome it should be as less as minimum*/). I have opted those model with whome I've earlier worked while Deep Learning.

1.The difference between model accuracy pre- and post-conversion was...
/*Based on the People Counted by each model*/
a>ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03: 0.53 | 0.27
b>ssdlite_mobilenet_v2_coco_2018_05_09: 0.61 | 0.44
c>ssd_mobilenet_v2_coco_2018_03_29: 0.71 | 0.57

2.The size of the model pre- and post-conversion was...
a>ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03: 44.4MB | 40.7MB
b>ssdlite_mobilenet_v2_coco_2018_05_09: 48.7MB | 45.32MB
c>ssd_mobilenet_v2_coco_2018_03_29: 179MB | 174.89MB

3.The inference time of the model pre- and post-conversion was...
a>ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03: 22ms | 28ms
b>ssdlite_mobilenet_v2_coco_2018_05_09: 26ms  | 31ms
c>ssd_mobilenet_v2_coco_2018_03_29: 63ms | 69ms
>person-detection-retail-0013: 45ms //Final Model Selected that provided Perfect Result ## Our Resultant Model proposed to be used for the required purpose





## Assess Model Use Cases
Some of the potential use cases of the people counter app are...
a. Amazon Go- In this era of social distancing 
b. Wallmart or other retail stores
c. Transport Stop at Hill station to count the number of tourist keeping the density small to safe it from becoming landslide prone
d. Office meeting rooms
e. Movie Theatres entrance for regulating covid19 precautions
f. Security fields like LOCs, Border Posts to examine the traffic passed perday 
g. Camera's near war-terrain will immediately count if any terrorist has tresspassed and will eventually help to capture as per counting
h. In flights, railways to ensure safe regulation during this pandemic era 

## Assess Effects on End User Needs
Lighting: Adverse Effect on counting-> Accuracy drops being directly proportional to Lighting condition -> Lightning dims and the accuracy will drop drastically,
model accuracy-> Best case possible accuracy is expected-> accuracy will directly define the global usage-> If model is of higher accuracy its acceptablity will grow among above mentioned use cases,  
camera focal length/image size have different effects on a deployed edge model-> For a broad retail- with smaller focal length/image size; much more area can be covered, counted & recognized( but key feature which defines focal length/image size is connected totally with accuracy)-> however with higher image size it will be extremely challenging for inferencing-> winding all total the focal length should neither be too zoomed in nor to far away it should be at a distance from which model can accurately count without loosing accuracy. 

