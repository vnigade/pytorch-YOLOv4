import argparse
from tool.torch_utils import do_detect
import torch
import torch.backends.cudnn as cudnn
from tool.darknet2pytorch import Darknet
import timeit
import os
import numpy as np
import GPUtil

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
    parser.add_argument('--model_config_dir', type=str,
                      default='cfg/', help='Model config directory')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Maximum batch size')
    parser.add_argument('--input_size', type=int,
                        help='Input size of model')
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

def test_model_time(opts, model, frame_size, total_iter=1):
    output_file = open('profile_latency_{}.txt'.format(frame_size), 'w')
    print("{:<20s},{:<20s},{:<20s}".format(
        "ModelSize", "Batch", "InferenceTime"), file=output_file)
    for batch in range(6,opts.max_batch_size+1,1):
        for _ in range(total_iter):
            # input = torch.rand([batch, frame_size, frame_size, 3])
            input = np.random.rand(batch, frame_size, frame_size, 3)
            print(input.nbytes)
            start_time = timeit.default_timer()
            with torch.no_grad():
               do_detect(model, input, 0.5, 0.4, use_cuda=(not opts.no_cuda))
                
            torch.cuda.synchronize()
            inference_time = (timeit.default_timer() - start_time) * 1000
            print("{:<20d},{:<20d},{:<20.2f}".format(
                frame_size, batch, inference_time), file=output_file)
            # torch.cuda.memory_summary(device=None, abbreviated=False)
            # GPUtil.showUtilization()
    output_file.close()

def load_model(opts, frame_size):
    # model = Yolov4(yolov4conv137weight=None,
    #               n_classes=opts.n_classes, inference=True)

    cfg_file_path = opts.model_config_dir + \
                  "/yolov4_" + str(frame_size) + ".cfg"
    model = Darknet(cfg_file_path, inference=True)
    weight_file = os.path.join(
        opts.weights_dir, "yolov4_{}.pth".format(frame_size))
    checkpoint = torch.load(
        weight_file, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    if not opts.no_cuda:
        model.cuda()

    cudnn.benchmarks = True
    cudnn.enabled = True

    # Zero grad for parameters
    for param in model.parameters():
        param.grad = None
    return model


if __name__ == "__main__":
    opts = parse_opts()

    # for frame_size in FRAME_SIZES:
    frame_size = opts.input_size
    model = load_model(opts, frame_size)
    test_model_time(opts, model, frame_size, total_iter=1)
