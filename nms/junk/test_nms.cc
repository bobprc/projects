#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <random>
#include <thrust/host_vector.h>

using namespace std;

thrust::host_vector<int> non_max_suppression_cuda(thrust::host_vector<float> &boxes,
                                  thrust::host_vector<int> &inds, 
                                  float thresh, int n_boxes, int batch_size);

int main()
{
  int nps = 6;
  int batch = 1;
  vector<vector<float>> boxes;
  vector<float> scores;
  default_random_engine e1(8);
  normal_distribution<float> jitter(0, 10);
  for(int b=0; b<batch; ++b)
  {
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
  }

  //cout << "...\n";
  //vector<int> mask = nms(boxes, scores, 0.9);
  //cout << "...\n";


  vector<int> inds(boxes.size());
  iota(inds.begin(), inds.end(), 0);
  sort(inds.begin(), inds.end(), [&scores](int a, int b){return scores[a] > scores[b];});



  thrust::host_vector<float> flat_boxes(4*boxes.size());
  thrust::host_vector<int> thinds(boxes.size());
  for(int i=0; i<boxes.size(); ++i)
  {
    for(int j=0; j<4; ++j)
    {
      flat_boxes[4*i+j]=boxes[inds[i]][j];
    }
    thinds[i] = inds[i];
  }
  cout << "...\n";
  thrust::host_vector<int> cu_mask = non_max_suppression_cuda(flat_boxes, thinds, 0.9, boxes.size(), 1);

  cout << "...\n";

/*  for(int i=0; i<mask.size(); ++i)
    unsorted_mask[inds[i]] = cu_mask[i];

  for(int i=0; i<mask.size(); ++i)
  {
    assert(unsorted_mask[i] == mask[i] || cout << i <<" " << mask[i] << "\n");
  }
*/
}
