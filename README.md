# Pytorch-YOLOv4

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [x] Train
    - [x] Mocaic

```
├── README.md
├── dataset.py       dataset
├── demo.py          demo to run pytorch --> tool/darknet2pytorch
├── darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_onnx.py     demo to run the converted onnx model
├── models.py        model for pytorch
├── train.py         train models.py
├── cfg.py           cfg.py for train
├── cfg              cfg --> darknet2pytorch
├── data            
├── weight           --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotatin.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

![image](https://user-gold-cdn.xitu.io/2020/4/26/171b5a6c8b3bd513?w=768&h=576&f=jpeg&s=78882)

# 0. Weights Download

## 0.1 darkent
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch
you can use darknet2pytorch to convert it yourself, or download my converted model.

- baidu
    - yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) 
    - yolov4.conv.137.pth(https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code:kcel)
- google
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)

# 1. Train

[use yolov4 to train your own data](Use_yolov4_to_train_your_own_data.md)

1. Download weight
2. Transform data

    For coco dataset,you can use tool/coco_annotatin.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
3. Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. Inference

- Load the pretrained darknet model and darknet weights to do the inference

```sh
python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
```

- Load pytorch weights (pth file) to do the inference

```sh
python models.py <num_classes> <weightfile> <imgfile> <namefile(optional)>
```


# 3. Darknet2ONNX (Evolving)

- **Pytorch version Recommended: 1.4.0**

- **Install onnxruntime**

    ```sh
    pip install onnxruntime
    ```

- **Run python script to generate onnx model and run the demo**

    ```sh
    python demo_onnx.py <cfgFile> <weightFile> <imageFile> <batchSize>
    ```

  This script will generate 2 onnx models.

  - One is for running the demo (batch_size=1)
  - The other one is what you want to generate (batch_size=batchSize)

# 4. ONNX2TensorRT (Evolving)

- **TensorRT version Recommended: 7.0, 7.1**

- **Run the following command to convert VOLOv4 onnx model into TensorRT engine**

    ```sh
    trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
    ```
    - Note: If you want to use int8 mode in conversion, extra int8 calibration is needed.

- **Run the demo (this demo here only works when batchSize=1)**

    ```sh
    python demo_trt.py <tensorRT_engine_file> <input_image> <input_H> <input_W>
    ```
    - Note1: input_H and input_W should agree with the input size in the original darknet cfg file as well as the latter onnx file.
    - Note2: extra NMS operations are needed for the tensorRT output. This demo uses TianXiaomo's NMS code from `tool/utils.py`.


# 5. ONNX2Tensorflow

- **First:Conversion to ONNX**

    tensorflow >=2.0
    
    1: Thanks:github:https://github.com/onnx/onnx-tensorflow
    
    2: Run git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    Run pip install -e .
    
    Note:Errors will occur when using "pip install onnx-tf", at least for me,it is recommended to use source code installation

Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```