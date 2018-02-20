import torch
import numpy as np
from torch.autograd import Variable
from test_utils import py_cpu_nms, torch_cuda_nms, generate_test_data
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--centroids", default=6, type=int)
parser.add_argument("--per-centroid", default=1000, type=int)
parser.add_argument("--batch-size", default=3, type=int)
parser.add_argument("--thresh", default=0.7, type=int)
parser.add_argument("--loops", default=20, type=int)
parser.add_argument("--sort", type=bool, default=False)
parser.add_argument("--test-pytorch-impl", action='store_true')


args = parser.parse_args()

if args.test_pytorch_impl:
    print('Chose to test PyTorch version, so setting batch size to 1 (was %i)' % args.batch_size)
    args.batch_size = 1

boxes_np, scores_np = generate_test_data(args.centroids, args.per_centroid, args.batch_size, args.sort)


t0 = datetime.now()
for _ in range(args.loops):
    [py_cpu_nms(np.concatenate((boxes_np[i], np.expand_dims(
     scores_np[i], -1)), -1), args.thresh) for i in range(args.batch_size)]
print("Python version took {} s to do {} loops".format((datetime.now() - t0).total_seconds(),
                                                       args.loops))

boxes = Variable(torch.Tensor(boxes_np))
scores = Variable(torch.Tensor(scores_np))

t0 = datetime.now()
for _ in range(args.loops):
    mask, inds = torch._C._VariableFunctions.non_max_suppression(boxes, scores, args.thresh)
print("C++ version took {} s to do {} loops".format((datetime.now() - t0).total_seconds(),
                                                    args.loops))


boxes = Variable(torch.Tensor(boxes_np)).cuda()
scores = Variable(torch.Tensor(scores_np)).cuda()


t0 = datetime.now()
for _ in range(args.loops):
    gpu_mask, gpu_inds = torch._C._VariableFunctions.non_max_suppression(boxes, scores, args.thresh)
print("C++ CUDA version took {} s to do {} loops".format((datetime.now() - t0).total_seconds(),
                                                  args.loops))

zeros_buffer = Variable(torch.cuda.LongTensor(scores_np.shape[1]).fill_(0))

if args.test_pytorch_impl:
    t0 = datetime.now()
    for _ in range(args.loops):
        inds = torch_cuda_nms(boxes.squeeze(0), scores.squeeze(0), args.thresh, zeros_buffer)
    print("PyTorch CUDA version took {} s to do {} loops".format(
            (datetime.now() - t0).total_seconds(), args.loops))
