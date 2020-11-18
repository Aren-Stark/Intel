#!/usr/bin/env python3
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class Network:
    def __init__(self):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device, size_in, size_out, num_req, cpu_extension=None, plugin=None):     
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
    
        if not plugin:
            self.plugin = IEPlugin(device = device)
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.net)
            not_supported_layers = \
                [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                sys.exit(1)
        
        if num_req == 0:
            self.net_plugin = self.plugin.load(network=self.net)
        else:
            self.net_plugin = self.plugin.load(network=self.net, num_req=num_req)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape
    
    
    def exec_net(self, request_id, frame):
        self.infer_request_handle = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        return self.net_plugin
        

    def wait(self, request_id):
        mutex = self.net_plugin.requests[request_id].wait(-1)
        return mutex
        
        
    def get_output(self, request_id, output=None):
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.net_plugin.requests[request_id].outputs[self.out_blob]
        return res
    