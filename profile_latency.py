import argparse

from numpy.core.fromnumeric import shape
from tool.torch_utils import do_detect
import torch
import torch.backends.cudnn as cudnn
from tool.darknet2pytorch import Darknet
import timeit, time
import os
import numpy as np
import cv2
import json
import random

FRAME_SIZES = [64,  96,  128, 160, 192, 224, 256,
               288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]

def set_deterministic_behaviour(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_opts():
    parser = argparse.ArgumentParser(description='Convert darknet model to pytorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--n_classes', type=int, default=80,
                        help='Number of classes in coco dataset')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--weights_dir', type=str,
                        help='Directory containing pytorch weights')
    parser.add_argument('--model_config_dir', type=str,
                        default='cfg/', help='Model config directory')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('--input_size', type=int,
                        help='Input size of model')
    parser.add_argument('--gt_annotations_path', type=str, default='instances_val2017.json',
                        help='ground truth annotations file')
    parser.add_argument('--dataset_dir', type=str,
                        default=None, help='dataset dir')
    parser.add_argument('--total_iter', type=int,
                        default=100, help='Total iterations')
    parser.add_argument('--log_dir', type=str, default="", help="Log dir")
    args = parser.parse_args()
    args_dict = args.__dict__
    print('{:-^100}'.format('Configurations'))
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print('{:-^100}'.format(''))

    return args


def read_image_in_jpg(opts, frame_size, index, batch_size, total_images, images):
    _ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    jpg_files = []
    indexes = []
    for _ in range(batch_size):
        # index = index % total_images
        index = random.randint(0, total_images-1)
        indexes.append(index)
        image_file_name = images[index]["file_name"]
        # print(image_file_name, opts.dataset_dir)
        img = cv2.imread(os.path.join(opts.dataset_dir, image_file_name))
        img = cv2.resize(img, (frame_size, frame_size), cv2.INTER_NEAREST)
        jpg_file = cv2.imencode(".jpg", img, _ENCODE_PARAM)[1].tobytes()
        jpg_files.append(jpg_file)
    print("Images: ", indexes)
    return jpg_files


def read_jpg_in_numpy(jpg_files, frame_size):
    imgs = []
    for jpg_file in jpg_files:
        img = cv2.imdecode(np.fromstring(
            jpg_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (frame_size, frame_size), cv2.INTER_NEAREST)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def test_model_time(opts, model, frame_size, annotations):
    output_file = open(f'{opts.log_dir}/profile_latency_{frame_size}.txt'.format(frame_size), 'w')
    print("{:<20s},{:<20s},{:<20s}".format(
        "ModelSize", "Batch", "InferenceTime"), file=output_file)
    images = annotations["images"]
    img_idx = 0
    total_images = len(images)
    for batch in range(1, opts.max_batch_size+1, 1):
        print("Processing batch size", batch)
        for _ in range(opts.total_iter):
            # time.sleep(0.004)
            jpg_files = read_image_in_jpg(
                opts, frame_size, img_idx, batch, total_images, images)

            # input = np.random.rand(batch, frame_size, frame_size, 3)
            torch.cuda.synchronize(opts.gpu_id)
            start_time = timeit.default_timer()
            start_perf = time.perf_counter()
            input = read_jpg_in_numpy(jpg_files, frame_size) # dtype is uint8
            batch_time = (timeit.default_timer() - start_time) * 1e3
            assert input.shape[0] == batch and input.shape[1] == frame_size \
                and input.shape[2] == frame_size and input.shape[3] == 3
            with torch.no_grad():
                output = do_detect(model, input, 0.5, 0.4,
                                   use_cuda=(not opts.no_cuda), gpu_number=opts.gpu_id)

            torch.cuda.synchronize(opts.gpu_id)
            inference_time = (timeit.default_timer() - start_time) * 1000
            perf_time = (time.perf_counter() - start_perf) * 1000
            print("Processing batch size:", batch, batch_time, inference_time, perf_time)
            print("{:<20d},{:<20d},{:<20.2f}".format(
                frame_size, batch, inference_time), file=output_file)
            # ms = torch.cuda.memory_summary(device=None, abbreviated=False)
            # stats = torch.cuda.memory_stats(device=None)
            # print(ms)
            # torch.cuda.empty_cache()
    output_file.close()


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
    annotations_file_path = opts.gt_annotations_path
    with open(annotations_file_path) as annotations_file:
        try:
            annotations = json.load(annotations_file)
        except:
            print("annotations file not a json")
            exit()

    set_deterministic_behaviour(1)
    frame_size = opts.input_size
    model = load_model(opts, frame_size)
    model.print_network()
    test_model_time(opts, model, frame_size, annotations)


