import argparse
import json
import os
import numpy as np
import torch
from tool.darknet2pytorch import Darknet
from tool.torch_utils import do_detect
import cv2
import torch.backends.cudnn as cudnn
import random


def set_deterministic_behaviour(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


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


def handle_output_det(det_boxes, frame_id, orig_image_width, orig_image_height):
    dets = []
    for det_box in det_boxes:
        x1 = round(det_box[0] * orig_image_width, 2)
        y1 = round(det_box[1] * orig_image_height, 2)
        x2 = round(det_box[2] * orig_image_width, 2)
        y2 = round(det_box[3] * orig_image_height, 2)
        w = round(x2 - x1, 2)
        h = round(y2 - y1, 2)
        bbox = [x1, y1, w, h]  # COCO format.
        det = {'image_id': frame_id,
               'category_id': int(det_box[-1]),
               'bbox': bbox,
               'score': round(float(det_box[-2]), 2),
               'image_height': orig_image_height,
               'image_width': orig_image_width
               }
        dets.append(det)

    return dets


def save_output_dets(log_dir, output_dets_json, frame_size):
    sorted_dets = list(sorted(output_dets_json, key=lambda x: x["image_id"]))
    sorted_dets = list(map(convert_cat_id_and_reorientate_bbox, sorted_dets))
    print(sorted_dets)
    dump_path = os.path.join(log_dir, f"model_{frame_size}")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    with open(os.path.join(dump_path, "output_dets.json"), 'w',
              encoding='utf-8') as file:
        json.dump(sorted_dets, file, indent=2)


def dump(model, opts, frame_size):
    output_dets_json = []
    cap = cv2.VideoCapture(opts.video_file)
    frame_id = 0
    while True:
        ret, img = cap.read()
        if ret is False:
            break
        orig_image_height, orig_image_width = img.shape[:2]
        # img = cv2.imread(os.path.join(cfg.dataset_dir, image_file_name))
        img = cv2.resize(
            img, (model.width, model.height), cv2.INTER_NEAREST)
        batch_boxes = do_detect(model, img, 0.20, 0.4, use_cuda=(
            not opts.no_cuda), gpu_number=opts.gpu_id)

        assert (len(batch_boxes) == 1)
        det_boxes = batch_boxes[0]
        dets = handle_output_det(
            det_boxes, frame_id, orig_image_width, orig_image_height)
        output_dets_json.extend(dets)

        frame_id += 1

    save_output_dets(opts.log_dir, output_dets_json, frame_size)


def parse_opts():
    parser = argparse.ArgumentParser(description='Dump object detections to file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--weights_dir', type=str,
                        help='Directory containing pytorch weights')
    parser.add_argument('--model_config_dir', type=str,
                        default='cfg/', help='Model config directory')
    parser.add_argument('--video_file', type=str,
                        help='video file')
    parser.add_argument('--input_size', type=str,
                        help='Input size of model')
    parser.add_argument('--log_dir', type=str, default="", help="Log dir")
    args = parser.parse_args()
    args_dict = args.__dict__
    print('{:-^100}'.format('Configurations'))
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print('{:-^100}'.format(''))

    return args


def load_model(opts, frame_size):
    cfg_file_path = opts.model_config_dir + \
        "/yolov4_" + str(frame_size) + ".cfg"
    model = Darknet(cfg_file_path, inference=True)
    weight_file = os.path.join(
        opts.weights_dir, "yolov4_{}.pth".format(frame_size))
    checkpoint = torch.load(
        weight_file, map_location='cuda:{}'.format(opts.gpu_id))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    if not opts.no_cuda:
        model.cuda(opts.gpu_id)

    # Zero grad for parameters
    for param in model.parameters():
        param.grad = None
    return model


if __name__ == "__main__":
    opts = parse_opts()
    torch.cuda.set_device(opts.gpu_id)

    frame_size = opts.input_size
    model = load_model(opts, frame_size)
    model.print_network()

    dump(model, opts, frame_size)
