"""
A script to evaluate the model's performance using pre-trained weights using COCO API.
Example usage: python evaluate_on_coco.py -dir D:\cocoDataset\val2017\val2017 -gta D:\cocoDataset\annotatio
ns_trainval2017\annotations\instances_val2017.json -c cfg/yolov4-smaller-input.cfg -g 0
Explanation: set where your images can be found using -dir, then use -gta to point to the ground truth annotations file
and finally -c to point to the config file you want to use to load the network using.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cfg import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names
from tool.torch_utils import do_detect
import cv2


def get_class_name(cat):
    class_names = load_class_names("./data/coco.names")
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return class_names[cat]


def convert_cat_id_and_reorientate_bbox(single_annotation):
    cat = single_annotation['category_id']
    bbox = single_annotation['bbox']
    x1, y1, w, h = bbox
    # x_center, y_center, w, h = bbox
    # x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11
    single_annotation['category_id'] = cat
    single_annotation['bbox'] = [x1, y1, w, h]  # COCO format
    return single_annotation


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj


def evaluate_on_coco(cfg, resFile):
    annType = "bbox"  # specify type here
    with open(resFile, 'r') as f:
        unsorted_annotations = json.load(f)
    sorted_annotations = list(sorted(
        unsorted_annotations, key=lambda single_annotation: single_annotation["image_id"]))
    sorted_annotations = list(
        map(convert_cat_id_and_reorientate_bbox, sorted_annotations))
    reshaped_annotations = defaultdict(list)
    for annotation in sorted_annotations:
        reshaped_annotations[annotation['image_id']].append(annotation)

    with open('temp.json', 'w') as f:
        json.dump(sorted_annotations, f)

    cocoGt = COCO(cfg.gt_annotations_path)
    cocoDt = cocoGt.loadRes('temp.json')

    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def test(model, annotations, cfg):
    if not annotations["images"]:
        print("Annotations do not have 'images' key")
        return
    images = annotations["images"]
    # images = images[:10]
    resFile = 'data/coco_val_outputs.json'

    if torch.cuda.is_available():
        use_cuda = 1
        model.cuda()
    else:
        use_cuda = 0

    # do one forward pass first to circumvent cold start
    # throwaway_image = Image.open('data/dog.jpg').convert('RGB').resize((model.width, model.height))
    throwaway_image = cv2.imread('data/dog.jpg')
    throwaway_image = cv2.resize(
        throwaway_image, (model.width, model.height), cv2.INTER_NEAREST)
    # do_detect(model, throwaway_image, 0.5, 0.4, use_cuda)
    do_detect(model, throwaway_image, 0.20, 0.4, use_cuda)

    total_images = len(images)
    boxes_json = []
    i = 0
    while i < total_images:
        batch_start = i
        batch_end = i + cfg.batch_size
        if batch_end > total_images:
            batch_end = total_images
        i = batch_end

        batch_images = []
        batch_image_annotations = {}
        for batch in range(batch_start, batch_end):
            image_annotation = images[batch]
            logging.info(
                "currently on image: {}/{}".format(batch + 1, len(images)))
            image_file_name = image_annotation["file_name"]
            image_id = image_annotation["id"]
            batch_image_annotations[batch] = image_annotation
            img = cv2.imread(os.path.join(cfg.dataset_dir, image_file_name))
            sized = cv2.resize(
                img, (model.width, model.height), cv2.INTER_LINEAR)
            # print("File Name: ", image_file_name)
            # print("Image size: (height, width) = ({}, {})".format(
            #    img.shape[0], img.shape[1]))
            # print("Sized Image size: (height, width) = ({}, {})".format(
            #    sized.shape[0], sized.shape[1]))
            batch_images.append(sized)

        batch_images = np.array(batch_images)
        # print("Batch images shape", batch_images.shape)
        start = time.time()
        with torch.no_grad():
            # batch_boxes = do_detect(model, batch_images, 0.5, 0.4, use_cuda)
            batch_boxes = do_detect(model, batch_images, 0.20, 0.4, use_cuda)
        finish = time.time()
        if type(batch_boxes) == list:
            assert len(batch_boxes) == (batch_end-batch_start)
            for b in range(len(batch_boxes)):
                batch = batch_start + b
                image_id = batch_image_annotations[batch]["id"]
                image_height = batch_image_annotations[batch]["height"]
                image_width = batch_image_annotations[batch]["width"]
                # print("Image id: ", image_id)
                for box in batch_boxes[b]:
                    box_json = {}
                    category_id = box[-1]
                    score = box[-2]
                    bbox_normalized = box[:4]
                    box_json["category_id"] = int(category_id)
                    box_json["image_id"] = int(image_id)

                    x1 = round(box[0] * image_width, 2)
                    y1 = round(box[1] * image_height, 2)
                    x2 = round(box[2] * image_width, 2)
                    y2 = round(box[3] * image_height, 2)
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]  # COCO format.
                    # print("Box in COCO: ", bbox)
                    box_json["bbox_normalized"] = list(
                        map(lambda x: round(float(x), 2), bbox_normalized))
                    box_json["bbox"] = bbox
                    box_json["score"] = round(float(score), 2)
                    box_json["timing"] = float(finish - start)
                    boxes_json.append(box_json)
                # print("see box_json: ", box_json)
                # print("See box: ", convert_cat_id_and_reorientate_bbox(box_json))
                    # json.dump(boxes_json, outfile, default=myconverter)
        else:
            print(
                "warning: output from model after postprocessing is not a list, ignoring")
            return

        # namesfile = 'data/coco.names'
        # class_names = load_class_names(namesfile)
        # plot_boxes(img, boxes, 'data/outcome/predictions_{}.jpg'.format(image_id), class_names)

    with open(resFile, 'w') as outfile:
        json.dump(boxes_json, outfile, default=myconverter)

    evaluate_on_coco(cfg, resFile)


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Test model on test dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-gta', '--ground_truth_annotations', type=str, default='instances_val2017.json',
                        help='ground truth annotations file', dest='gt_annotations_path')
    parser.add_argument('-w', '--weights_file', type=str, default='weights/yolov4.weights',
                        help='Pytorch weights file to load', dest='weights_file')
    parser.add_argument('-c', '--model_config', type=str, default='cfg/yolov4.cfg',
                        help='model config file to load', dest='model_config')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size ', dest='batch_size')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime

    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = Darknet(cfg.model_config, inference=True)
    model.print_network()

    checkpoint = torch.load(
        cfg.weights_file, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_weights(cfg.weights_file)

    model.eval()  # set model away from training

    # if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)
    model.to(device=device)

    annotations_file_path = cfg.gt_annotations_path
    with open(annotations_file_path) as annotations_file:
        try:
            annotations = json.load(annotations_file)
        except:
            print("annotations file not a json")
            exit()
    test(model=model,
         annotations=annotations,
         cfg=cfg, )
