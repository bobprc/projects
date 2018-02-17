#include "ATen/NativeFunctions.h"

#include <tuple>

namespace at {
namespace native {

__device__ __forceinline__ float fmin(float a, float b)
{
  return a > b ? b : a;
}

__device__ __forceinline__ float fmax(float a, float b)
{
  return a > b ? a : b;
}

__device__ __forceinline__ int elim_idx(int x, int y, int n)
{
  // Calculate an index into the boolean array "elims". The
  // array specifies which boxes are to be eliminated:
  // elims[elim_idx(x, y, n)] = 0 if box x eliminates box y,
  // 1 otherwise. n is the total number of boxes.
  return x <= (n-1)/2? x*(n-1) + y: n*(n-2-x) + y+1;
}

template <typename T>
__global__ void nms_elims_kernel(
  const T *boxes,
  const int64_t *sorted_idx,
  unsigned char *elims,
  const float thresh,
  const int64_t num_boxes)
{
  // For each pair of boxes, determine whether the higher-scoring box
  // eliminates the lower-scoring one. Store this information in the array
  // "elims". Each thread handles a pair of boxes.

  // Use a flat index so that we have only batch_size underutilised blocks.
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx >= (num_boxes*(num_boxes-1))/2) // number of pairs
    return;

  // Calculate the indices of the two boxes processed by this thread. The
  // mapping idx -> box_x_idx, box_y_idx is chosen so as to avoid the use
  // of sqrtf or potentially long loops in the calculation.
  int base = idx / num_boxes;
  int midpt = (base+1) * (num_boxes-1);
  int box_x_idx, box_y_idx;
  if(idx >= midpt)
  {
    box_x_idx = num_boxes - base - 2 + num_boxes * blockIdx.y;
    box_x_idx = sorted_idx[box_x_idx];
    box_y_idx = idx - midpt + box_x_idx + 1 + num_boxes * blockIdx.y;
    box_y_idx = sorted_idx[box_y_idx];
  }
  else
  {
    box_x_idx = base + num_boxes * blockIdx.y;
    box_x_idx = sorted_idx[box_x_idx];
    box_y_idx = idx - midpt + num_boxes + num_boxes * blockIdx.y;
    box_y_idx = sorted_idx[box_y_idx];
  }

  float box_x[4], box_y[4];

  for(int i=0; i<4; ++i)
  {
    box_x[i] = boxes[box_x_idx*4 + i];
    box_y[i] = boxes[box_y_idx*4 + i];
  }
  
  // Calculate IoU between the boxes.
  float rightmost_l = fmax(box_x[0], box_y[0]);
  float leftmost_r = fmin(box_x[0] + box_x[2], box_y[0] + box_y[2]);
  float delta_x = fmax(0., leftmost_r - rightmost_l);

  float bottommost_tp = fmax(box_x[1], box_y[1]);
  float topmost_b = fmin(box_x[1] + box_x[3], box_y[1] + box_y[3]);
  float delta_y = fmax(0., topmost_b - bottommost_tp);

  float uni = box_x[2] * box_x[3] + box_y[2] * box_y[3];

  float iou = delta_x * delta_y / (uni - delta_x * delta_y);

  // Write 0 if box x eliminates box y.
  int batch_shift = (num_boxes*(num_boxes-1))/2*blockIdx.y;
  if (iou > thresh)
    elims[elim_idx(box_x_idx, box_y_idx, num_boxes) + batch_shift] = 0;

}

__global__ void nms_mask_kernel(unsigned char *mask, unsigned char *elims, int64_t num_boxes)
{
  // Given "elims", calculate the mask. Unfortunately this
  // requires global synchronisation, so launch only one block
  // per batch element and use a for loop to cover all the boxes.
  int col = 0;
  int batch_shift = (num_boxes*(num_boxes-1))/2*blockIdx.x;
  while(col < num_boxes-1)
  {
    for(int i = threadIdx.x; i < num_boxes-1; i+=blockDim.x)
      if(i >= col)
        mask[i+1+blockIdx.x*num_boxes] *= elims[elim_idx(col, i, num_boxes)+batch_shift];
    __syncthreads();
    ++col;
    while((col < num_boxes - 1) && mask[col+blockIdx.x*num_boxes]==0)
      ++col;
  }
}

std::tuple<Tensor, Tensor> non_max_suppression_cuda(const Tensor& input, const Tensor& scores, const double thresh)
{

  AT_ASSERT(input.ndimension() == 3, "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_ASSERT(scores.ndimension() == 2, "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_ASSERT(input.size(0) == scores.size(0), "First and second arguments must have equal-sized first dimension");
  AT_ASSERT(input.size(1) == scores.size(1), "First and second arguments must have equal-sized second dimension");
  AT_ASSERT(input.size(2) == 4, "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");


  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});
  mask.fill_(1);
  int n_pairs = (num_boxes*(num_boxes-1))/2;
  
  unsigned char *elims;
  cudaMalloc(&elims, n_pairs*batch_size*sizeof(bool));
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "Failed to allocate memory for non_max_suppression");

  //need the indices of the boxes sorted by score.
  Tensor sorted_inds = std::get<1>(scores.sort(-1, true));


  dim3 elims_block(512);
  dim3 elims_grid((n_pairs-1+512)/512, batch_size);
  nms_elims_kernel<<<elims_grid, elims_block, 0, globalContext().getCurrentCUDAStream()>>>(
                                                 input.data<float>(),
                                                 sorted_inds.data<int64_t>(),
                                                 elims, 
                                                 thresh,
                                                 num_boxes);
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "nms_elims_kernel failed");
  dim3 mask_block(512);
  dim3 mask_grid(batch_size);
  nms_mask_kernel<<<mask_grid, mask_block, 0, globalContext().getCurrentCUDAStream()>>>(
                                    mask.data<unsigned char>(),
                                    elims,
                                    num_boxes);
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "nms_mask_kernel failed");

  //It's not entirely clear what the best thing to return is here. The algorithm will
  //produce a different number of boxes for each batch, so there is no obvious way of
  //way of returning the surving boxes/indices as a tensor. Returning a mask on the
  //sorted boxes together with the sorted indices seems reasonable; that way, the user
  //can easily take the N highest-scoring surviving boxes to form a tensor if they wish. 
  return std::make_tuple(mask, sorted_inds);
}

}}
