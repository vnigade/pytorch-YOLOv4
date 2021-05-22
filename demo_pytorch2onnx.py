import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from tool.darknet2pytorch import Darknet

from tool.utils import *
from models import Yolov4
from demo_darknet2onnx import detect

def load_model(model_config_file, weight_file, frame_size):
    model = Darknet(model_config_file, inference=True)
    checkpoint = torch.load(
        weight_file, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    model.cuda()
    return model


def transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W, model_config_file):
    
    # model = Yolov4(n_classes=n_classes, inference=True)

    # pretrained_dict = torch.load(weight_file, map_location=torch.device('cuda'))
    # model.load_state_dict(pretrained_dict)
    model = load_model(model_config_file, weight_file, frame_size=IN_IMAGE_H)

    input_names = ["input"]
    output_names = ['boxes', 'confs']

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name
    


def main(weight_file, image_path, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W, model_config_file):

    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W, model_config_file)
    else:
        # Transform to onnx as specified batch size
        transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo = transform_to_onnx(weight_file, 1, n_classes, IN_IMAGE_H, IN_IMAGE_W)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    image_src = cv2.imread(image_path)
    detect(session, image_src)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 8:
        
        weight_file = sys.argv[1]
        image_path = sys.argv[2]
        batch_size = int(sys.argv[3])
        n_classes = int(sys.argv[4])
        IN_IMAGE_H = int(sys.argv[5])
        IN_IMAGE_W = int(sys.argv[6])
        model_config_file = sys.argv[7]

        main(weight_file, image_path, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W, model_config_file)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <weight_file> <image_path> <batch_size> <n_classes> <IN_IMAGE_H> <IN_IMAGE_W>')
