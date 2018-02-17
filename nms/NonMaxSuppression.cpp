Tensor NonMaxSupression_forward(const Tensor& input,
                                const Tensor& scores,
                                const float thresh)
{

  AT_ASSERT(input.ndimension() == 3, "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_ASSERT(scores.ndimension() == 4, "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_ASSERT(input.size(0) == scores.size(0), "First and second arguments must have equal-sized first dimensions");
  AT_ASSERT(input.size(1) == scores.size(1), "First and second arguments must have equal-sized second dimensions");
  AT_ASSERT(input.size(2) == 4, "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");
  std::tuple<Tensor, Tensor> scores_and_inds = scores.sort(-1, true);
  
  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});

  auto *rawInput = input.data<float>();
  auto *rawMask = mask.data<bool>();
  auto *rawIdx = std::get<1>(scores_and_inds).data<int>();

  for(int batch=0; batch<batch_size; ++batch){
    int pos=batch*num_boxes;
    while(pos < num_boxes-1)
    {
      for(int i=pos+1; i<num_boxes*(1+batch); ++i)
      {
        int idx_x = rawIdx[pos];
        int idx_y = rawIdx[i];
        float lr = std::fmin(input[idx_x] + input[idx_x+2], 
                             input[idx_y] + input[idx_y+2]);
        float rl = std::fmax(input[idx_x], input[idx_y]);
        float tb = std::fmin(input[idx_x+1] + input[idx_x+3],
                             input[idx_y+1] + input[idx_y+3]);
        float bt = std::fmax(input[idx_x+1],  input[idx_y+1]);
        float inter = std::fmax(0, lr-rl)*std::fmax(0, tb-bt);
        float uni = (input[idx_x+2]*input[idx_x+3] 
                     + input[idx_y+2]*input[idx_y+3]
                     - inter);
        if(inter/uni > thresh)
          mask[idx_y] = 0;
      }
      ++pos;
      while(rawIdx[pos] < num_boxes-1 and mask[rawIdx[pos]] == 0)
        ++pos;
    }

  return std::make_tuple(mask, std::get<1>(scores_and_inds));
}
