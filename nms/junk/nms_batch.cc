#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

std::vector<int> nms(const std::vector<std::vector<float> > & boxes,
                     const std::vector<int> & inds,
                     const std::vector<float> & scores,
                     const float thresh)
{
  const int size = boxes.size();

  std::vector<int> mask(size, 1);
  int posp=0;
  while(posp < size-1)
  {
    int pos = inds[posp];
    for(int ii=pos+1; i<size; ++i)
    {
      int i = inds[ii];
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
        mask[ii] = 0;
    }
    ++posp;
    while(posp < size-1 and mask[posp] == 0)
      ++posp;
  }
  std::vector<int> unsorted_mask(size);
  for(int i=0; i<size; ++i)
    unsorted_mask[indices[i]]=mask[i];

  return unsorted_mask;
}
