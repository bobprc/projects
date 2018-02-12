#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <random>
#include <thrust/host_vector.h>

using namespace std;

vector<int> nms(const vector<vector<float>>&,
                const vector<float>&,
                const float);

thrust::host_vector<int> nms_cuda(thrust::host_vector<float> &boxes,
                                  float thresh, int n_boxes);

int main()
{
  int nps = 6000;
  vector<vector<float>> boxes;
  vector<float> scores;
  default_random_engine e1(8);
  normal_distribution<float> jitter(0, 10);
  for(int i=0; i<nps; ++i)
  {
    vector<float> box{150 + jitter(e1),
                      200 + jitter(e1),
                      30 + jitter(e1),
                      100 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }
  for(int i=0; i<nps; ++i)
  {
    vector<float> box{1000 + jitter(e1),
                      560 + jitter(e1),
                      300 + jitter(e1),
                      100 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }
  for(int i=0; i<nps; ++i)
  {
    vector<float> box{100 + jitter(e1),
                      2000 + jitter(e1),
                      300 + jitter(e1),
                      200 + jitter(e1)};
    boxes.push_back(box);
    scores.push_back(jitter(e1));
  }

  cout << "...\n";
  vector<int> mask = nms(boxes, scores, 0.9);
  cout << "...\n";


  vector<int> inds(boxes.size());
  iota(inds.begin(), inds.end(), 0);
  sort(inds.begin(), inds.end(), [&scores](int a, int b){return scores[a] > scores[b];});



  thrust::host_vector<float> flat_boxes(4*boxes.size());
  for(int i=0; i<boxes.size(); ++i)
  {
    for(int j=0; j<4; ++j)
    {
      flat_boxes[4*i+j]=boxes[inds[i]][j];
    }
  }
  cout << "...\n";
  thrust::host_vector<int> cu_mask = nms_cuda(flat_boxes, 0.9, boxes.size());
  cout << "...\n";
  vector<int> unsorted_mask(mask.size());

  for(int i=0; i<mask.size(); ++i)
    unsorted_mask[inds[i]] = cu_mask[i];

  for(int i=0; i<mask.size(); ++i)
  {
    assert(unsorted_mask[i] == mask[i] || cout << i <<" " << mask[i] << "\n");
  }

}
