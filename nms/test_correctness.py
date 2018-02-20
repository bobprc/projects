import torch
import numpy as np
from torch.autograd import Variable
from test_utils import generate_test_data, py_cpu_nms
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--centroids", default=6, type=int)
parser.add_argument("--per-centroid", default=1000, type=int)
parser.add_argument("--batch-size", default=4, type=int)
parser.add_argument("--thresh", default=0.7, type=int)

args = parser.parse_args()

boxes_np, scores_np = generate_test_data(args.centroids, args.per_centroid, args.batch_size)


boxes = Variable(torch.Tensor(boxes_np)).cuda()
scores = Variable(torch.Tensor(scores_np)).cuda()

gpu_mask, gpu_inds = torch._C._VariableFunctions.non_max_suppression(boxes, scores, args.thresh)

pyres = []
for i in range(args.batch_size):
    pyres.append(py_cpu_nms(np.concatenate(
                (boxes_np[i], np.expand_dims(scores[i], -1)), -1), args.thresh))


boxes = Variable(torch.Tensor(boxes_np))
scores = Variable(torch.Tensor(scores_np))

mask, inds = torch._C._VariableFunctions.non_max_suppression(boxes, scores, args.thresh)

for i in range(args.batch_size):
    assert np.array_equal(pyres[i], inds[i][mask[i]].cpu().data.numpy())
    assert np.array_equal(pyres[i], gpu_inds[i][gpu_mask[i]].cpu().data.numpy())
print("OK")
