import argparse
from tool.torch_utils import do_detect
import torch
from models import Yolov4
import timeit
import os

FRAME_SIZES = [64,  96,  128, 160, 192, 224, 256,
               288, 320, 352, 384, 416, 448, 480,
               512, 544, 576, 608, 640]


def parse_opts():
    parser = argparse.ArgumentParser(description='Convert darknet model to pytorch',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_classes', type=int, default=80,
                        help='Number of classes in coco dataset')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--weights_dir', type=str,
                        help='Directory containing pytorch weights')
    parser.add_argument('--max_batch_size', type=str,
                        help='Maximum batch size')
    # parser.add_argument('--model_config_dir', type=str,
    #                     default='cfg/', help='Model config directory')
    # parser.add_argument('--dst_dir', type=str,
    #                     default="pytorch_models/", help='Destination directory to save pytorch models')

    args = parser.parse_args()
    args_dict = args.__dict__
    print('{:-^100}'.format('Configurations'))
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print('{:-^100}'.format(''))

    return args


def test_model_time(opts, model, frame_size):
    print("\t{:<20s}{:<20s}{:<20s}".format(
        "ModelSize", "Batch", "InferenceTime"))
    for batch in range(opts.max_batch_size):
        for _ in range(100):
            input = torch.rand([batch, frame_size, frame_size, 3])
            start_time = timeit.default_timer()
            with torch.no_grad():
                _ = do_detect(model, input, 0.4, 0.5, use_cuda=opts.use_cuda)
            torch.cuda.synchronize()
            inference_time = (timeit.default_timer() - start_time) * 1000
            print("\t{:<20d}{:<20d}{:<20.2f}".format(
                "ModelSize", "Batch", "InferenceTime"))


def load_model(opts, frame_size):
    model = Yolov4(yolov4conv137weight=None,
                   n_classes=opts.n_classes, inference=True)

    weight_file = os.path.join(
        opts.weight_dir, "yolov4_{}.pth".format(frame_size))
    pretrained_dict = torch.load(
        weight_file, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    if opts.use_cuda:
        model.cuda()

    return model


if __name__ == "__main__":
    opts = parse_opts()

    for frame_size in FRAME_SIZES:
        model = load_model(opts, frame_size)
        test_model_time(model)
