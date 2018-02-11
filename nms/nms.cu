#include <cmath>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>

__device__ __forceinline__ int elim_idx(int x, int y, int n_bxs)
{
  return x*n_bxs - (x*(x+3))/2 + y - 1;
}

__global__ void nms_kern(float* boxes, int* elims, int* mask,
                       float thresh, int n_bxs)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx >= (n_bxs*(n_bxs-1))/2)
    return;


  int box_x_idx = n_bxs - static_cast<int>(ceil(0.5 + sqrtf((n_bxs-0.5)*(n_bxs-0.5)-2*idx)));
  int box_y_idx = idx + 1 - (n_bxs-1)*box_x_idx + ((box_x_idx+1)*box_x_idx)/2;

  float box_x[4], box_y[4];

  for(int i=0; i<4; ++i)
  {
    box_x[i] = boxes[box_x_idx*4 + i];
    box_y[i] = boxes[box_y_idx*4 + i];
  }

  float rightmost_l = fmax(box_x[0], box_y[0]);
  float leftmost_r = fmin(box_x[0] + box_x[2], box_y[0] + box_y[2]);
  float delta_x = fmax(0., leftmost_r - rightmost_l);

  float bottommost_tp = fmax(box_x[1], box_y[1]);
  float topmost_b = fmin(box_x[1] + box_x[3], box_y[1] + box_y[3]);
  float delta_y = fmax(0., topmost_b - bottommost_tp);

  float uni = box_x[2] * box_x[3] + box_y[2] * box_y[3];

  float iou = delta_x * delta_y / (uni - delta_x * delta_y);

 
  elims[idx] = (iou > thresh) ? 0 : 1;

  __syncthreads();

  if(box_x_idx != 0) 
    return;


  int col = 0;
  while(col < box_y_idx)
  {
    mask[box_y_idx] *= elims[elim_idx(col, box_y_idx, n_bxs)];
    __syncthreads();
    ++col;
    while(col < n_bxs - 1 && mask[col] == 0)
      ++col;
  }

}

thrust::host_vector<int> nms_cuda(thrust::host_vector<float> &cpu_boxes, float thresh, int n_boxes)
{
  cudaError_t err;
  thrust::device_vector<float> boxes = cpu_boxes;
  thrust::device_vector<int> mask(n_boxes, 1);
  thrust::device_vector<int> elims((n_boxes*(n_boxes-1))/2, 1);
  nms_kern<<<n_boxes, n_boxes>>>(thrust::raw_pointer_cast(boxes.data()),
                                        thrust::raw_pointer_cast(elims.data()), 
                                        thrust::raw_pointer_cast(mask.data()),
                                        thresh, n_boxes);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  thrust::host_vector<int> output = mask;
  return output;
}
