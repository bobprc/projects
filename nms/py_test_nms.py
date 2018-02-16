import numpy as np
from py_cpu_nms import py_cpu_nms
from non_max_s import non_max_s

def generate_test_data():
    boxes, scores = [], []
    centroids = np.random.choice(np.arange(200, 1000), size=[6, 2])
    for centroid in centroids:
        for _ in range(4000):
            boxes.append(list(np.concatenate((centroid + np.random.choice(np.arange(0, 30)),
                                              np.random.choice(np.arange(100, 200), size=2)))))
            scores.append(np.abs(np.random.normal()))
    return boxes, scores

boxes, scores = generate_test_data()

mine = np.nonzero(non_max_s(boxes, scores, 0.99))[0]

boxes = np.array(boxes)
scores = np.expand_dims(np.array(scores), -1)

dets = np.concatenate((boxes, scores), axis=-1)

his = py_cpu_nms(dets, 0.99)

his = np.sort(his)


assert np.array_equal(mine, his), "There was a discrepancy"

print("OK")

