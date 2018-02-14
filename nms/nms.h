#include <vector>

std::vector<int> nms(const std::vector<std::vector<float> > & boxes,
                     const std::vector<float> & scores,
                     const float thresh);
