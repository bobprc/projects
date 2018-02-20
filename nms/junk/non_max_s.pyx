from libcpp.vector cimport vector

cdef extern from "nms.h":
    vector[int] nms(vector[vector[float]], vector[float], float)

def non_max_s(boxes, scores, thresh):
    return nms(boxes, scores, thresh)
