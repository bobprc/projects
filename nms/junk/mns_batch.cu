#include <thrust/device_vector.h>

__device__ __forceinline__ int elim_idx(int x, int y, int n)
{
  // Calculate an index into the boolean array "elims". The
  // array specifies which boxes are to be eliminated:
  // elims[elim_idx(x, y, n)] = 0 if box x eliminates box y,
  // 1 otherwise. n is the total number of boxes.
  return x <= (n-1)/2? x*(n-1) + y-1: n*(n-2-x) + y;
}

template <typename T>
__global__ void nms_elims_kernel(
  const T *boxes,
  const int *sorted_idx,
  unsigned char *elims,
  const float thresh,
  const int num_boxes)
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
    box_y_idx = idx - midpt + box_x_idx + 1 + num_boxes * blockIdx.y;
    box_x_idx = sorted_idx[box_x_idx] + num_boxes * blockIdx.y;
    box_y_idx = sorted_idx[box_y_idx] + num_boxes * blockIdx.y;
  }
  else
  {
    box_x_idx = base + num_boxes * blockIdx.y;
    box_y_idx = idx - midpt + num_boxes + num_boxes * blockIdx.y;
    box_x_idx = sorted_idx[box_x_idx] + num_boxes * blockIdx.y;
    box_y_idx = sorted_idx[box_y_idx] + num_boxes * blockIdx.y;
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
  elims[idx + batch_shift] = (iou > thresh) ? 0 : 1;

}

__global__ void nms_mask_kernel(unsigned char *mask, unsigned char *elims, int num_boxes)
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
        mask[i+1+blockIdx.x*num_boxes] *= elims[elim_idx(col, i+1, num_boxes)+batch_shift];
    __syncthreads();
    ++col;
    while((col < num_boxes - 1) && (mask[col+blockIdx.x*num_boxes]==0))
      ++col;
  }
}

thrust::host_vector<int> non_max_suppression_cuda(thrust::host_vector<float> &input, thrust::host_vector<int> &inds, const float thresh, int n_boxes, int batch_size)
{


  int n_pairs = (n_boxes*(n_boxes-1))/2;
  thrust::device_vector<float> boxes = input;
  thrust::device_vector<int> sinds = inds;
  thrust::device_vector<unsigned char> mask(n_boxes*batch_size);
  thrust::device_vector<unsigned char> elims(n_pairs*batch_size);
  
  dim3 elims_block(512);
  dim3 elims_grid((n_pairs-1+512)/512, batch_size);
  nms_elims_kernel<<<elims_grid, elims_block>>>(
                                                    thrust::raw_pointer_cast(boxes.data()),
                                                    thrust::raw_pointer_cast(sinds.data()),
                                                    thrust::raw_pointer_cast(elims.data()),
                                                    thresh,
                                                    n_boxes);
  dim3 mask_block(512);
  dim3 mask_grid(batch_size);
  nms_mask_kernel<<<mask_grid, mask_block>>>(
                                    thrust::raw_pointer_cast(mask.data()),
                                    thrust::raw_pointer_cast(elims.data()),
                                    n_boxes);

  //It's not entirely clear what the best thing to return is here. The algorithm will
  //produce a different number of boxes for each batch, so there is no obvious way of
  //way of returning the surving boxes/indices as a tensor. Returning a mask on the
  //sorted boxes together with the sorted indices seems reasonable; that way, the user
  //can easily take the N highest-scoring surviving boxes to form a tensor if they wish. 
  thrust::host_vector<int> output = mask;
  return output;
}

