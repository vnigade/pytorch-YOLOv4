import argparse
from tool.darknet2pytorch import Darknet
import torch
import os

FRAME_SIZES = [64,  96,  128, 160, 192, 224, 256,
               288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]


def parse_opts():
    parser = argparse.ArgumentParser(description='Convert darknet model to pytorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--weights_file', type=str, default='weights/yolov4.weights',
                        help='YOLOv4 weights file to load')
    parser.add_argument('--model_config_dir', type=str,
                        default='cfg/', help='Model config directory')
    parser.add_argument('--dst_dir', type=str,
                        default="pytorch_models/", help='Destination directory to save pytorch models')

    args = parser.parse_args()
    args_dict = args.__dict__
    print('{:-^100}'.format('Configurations'))
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print('{:-^100}'.format(''))

    return args


def convert(opts, frame_size):
    cfg_file_path = opts.model_config_dir + \
        "/yolov4_" + str(frame_size) + ".cfg"
    model = Darknet(cfg_file_path)
    model.print_network()
    model.load_weights(opts.weights_file)

    if not opts.no_cuda:
        model = model.eval().cuda()

    state_dict = model.state_dict()

    states = {
        "frame_size": frame_size,
        "state_dict": state_dict
    }

    save_file_path = os.path.join(
        opts.dst_dir, 'yolov4_{}.pth'.format(frame_size))
    torch.save(states, save_file_path)


if __name__ == "main":
    opts = parse_opts()

    for frame_size in FRAME_SIZES:
        convert(opts, frame_size)
