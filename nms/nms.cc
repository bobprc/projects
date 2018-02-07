#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

std::vector<int> nms(const std::vector<std::vector<float>> & boxes,
                     const std::vector<float> & scores,
                     const float thresh)
{
  const int size = boxes.size();
  std::vector<int> indices(size);
  std::vector<std::vector<float>> sorted_boxes(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&scores](int a, int b){
     return scores[a] > scores[b];});
  for(int i=0; i<size; ++i)
  {
    sorted_boxes[i] = boxes[indices[i]];
  }
  
  std::vector<int> mask(size, 1);
  int pos=0;
  while(pos < size-1)
  {
    for(int i=pos+1; i<size; ++i)
    {
      float lr = std::fmin(sorted_boxes[pos][0] + sorted_boxes[pos][2], 
                           sorted_boxes[i][0] + sorted_boxes[i][2]);
      float rl = std::fmax(sorted_boxes[pos][0], sorted_boxes[i][0]);
      float tb = std::fmin(sorted_boxes[pos][1] + sorted_boxes[pos][3],
                           sorted_boxes[i][1] + sorted_boxes[i][3]);
      float bt = std::fmax(sorted_boxes[pos][1],  sorted_boxes[i][1]);
      float inter = std::fmax(0, lr-rl)*std::fmax(0, tb-bt);
      float uni = (sorted_boxes[pos][2]*sorted_boxes[pos][3] 
                   + sorted_boxes[i][2]*sorted_boxes[i][3]
                   - inter);
      if(inter/uni > thresh)
        mask[i] = 0;
    }
    ++pos;
    while(pos < size-1 and mask[pos] == 0)
    {
    ++pos;
    }
  }
  std::vector<int> unsorted_mask(size);
  for(int i=0; i<size; ++i)
  {
    unsorted_mask[indices[i]]=mask[i];
  }
  return unsorted_mask;
}
