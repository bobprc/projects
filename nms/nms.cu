#include <cmath>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>

__device__ __forceinline__ int elim_idx(int x, int y, int n)
{
  return x <= (n-1)/2? x*(n-1) + y: n*(n-2-x) + y+1;
}


__global__ void calc_elims(float* boxes, int* elims, float thresh, int n_bxs)
{

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if(idx >= (n_bxs*(n_bxs-1))/2)
    return;

  int base = idx / n_bxs;
  int midpt = (base+1) * (n_bxs-1);
  int box_x_idx, box_y_idx;
  if(idx >= midpt)
  {
    box_x_idx = n_bxs-base-2;
    box_y_idx = idx - midpt + box_x_idx + 1;
  }
  else
  {
    box_x_idx = base;
    box_y_idx = idx - midpt + n_bxs;
  }


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
}

__global__ void calc_mask(int* elims, int* mask, int n_bxs)
{
  int col = 0;
  while(col < n_bxs-1)
  {
    for(int i = threadIdx.x; i < n_bxs-1; i+=blockDim.x)
      if(i >= col)
        mask[i+1] *= elims[elim_idx(col, i, n_bxs)];
    __syncthreads();
    ++col;
    while((col < n_bxs - 1) && (mask[col] == 0))
      ++col;
  }
}

thrust::host_vector<int> nms_cuda(thrust::host_vector<float> &cpu_boxes, float thresh, int n_boxes)
{
  int n_pairs = (n_boxes*(n_boxes-1))/2;
  cudaError_t err;
  thrust::device_vector<float> boxes = cpu_boxes;
  thrust::device_vector<int> mask(n_boxes, 1);
  thrust::device_vector<int> elims(n_pairs);

  calc_elims<<<(n_pairs-1+512)/512, 512>>>(thrust::raw_pointer_cast(boxes.data()),
                                         thrust::raw_pointer_cast(elims.data()), 
                                         thresh, n_boxes);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "calc elims failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  calc_mask<<<1, 512>>>(thrust::raw_pointer_cast(elims.data()), 
                        thrust::raw_pointer_cast(mask.data()),
                        n_boxes);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "calc mask failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  cudaDeviceSynchronize();
  thrust::host_vector<int> output = mask;
  return output;
}
