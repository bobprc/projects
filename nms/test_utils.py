# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def torch_cuda_nms(dets, scores, thresh, ind_buffer):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + x1
    y2 = dets[:, 3] + y1

    scores = scores
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(dim=0, descending=True)
    i = 0
    while order.size()[0] > 1:
        ind_buffer[i] = order[0]
        i += 1
        xx1 = torch.max(x1[order[0]], x1[order[1:]])
        yy1 = torch.max(y1[order[0]], y1[order[1:]])
        xx2 = torch.min(x2[order[0]], x2[order[1:]])
        yy2 = torch.min(y2[order[0]], y2[order[1:]])

        w = F.relu(xx2 - xx1)
        h = F.relu(yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = torch.nonzero(ovr <= thresh).squeeze()
        if inds.dim():
            order = order[(inds + 1)]
        else:
            break
    keep = ind_buffer[:i]
    return keep

import numpy as np

def generate_test_data(cnt, npc, batch_size, sort=False):
    box_batches, score_batches = [], []
    for _ in range(batch_size):
        boxes, scores = [], []
        centroids = np.random.choice(np.arange(200, 1000), size=[cnt, 2])
        for centroid in centroids:
            for _ in range(npc):
                boxes.append(list(np.concatenate((centroid + np.random.choice(np.arange(0, 30)),
                                                  np.random.choice(np.arange(100, 200), size=2)))))
                scores.append(np.abs(np.random.normal()))
        if sort:
            inds = np.argsort(scores, -1)
            box_batches.append(np.array(boxes)[inds])
            score_batches.append(np.array(scores)[inds])
        else:
            box_batches.append(np.array(boxes))
            score_batches.append(np.array(scores))
    return np.array(box_batches).astype(np.float32), np.array(score_batches)

