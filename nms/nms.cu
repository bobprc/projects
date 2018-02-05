#include <cmath>


__device__ __forceinline__ int elim_idx(int x, int y, int n_bxs)
{
  return x*n_bxs - (x*(x-1))/2 + y - 1;
}

__global__ void nms_kern(float* boxes, int* elims, int* mask,
                       float thresh, int n_bxs)
{

int idx = threadIdx.x + blockDim.x * blockIdx.x;

if(idx >= (n_bxs *  (n_bxs - 1))/2)
{
  return;
}

int box_x_idx = n_bxs - static_cast<int>(sqrt((n-1/2)*(n-1/2)-2*idx));
int box_y_idx = idx + 1 + (box_x_idx * (box_x_idx+1))/2 - box_x_idx * (n_bxs-1);

float box_x[4], box_y[4];

for(int i=0; i<4; i++)
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

if(iou < thresh)
{
  elims[idx] = 1;
}

__syncthreads();

if(box_x_idx != 0)
{
  return;
}

int col = 0, row = 0;
while(col < box_y_idx)
{
  mask[box_y_idx] *= elims[elim_idx(col, box_y_idx, n_bxs)];
  while(row + col < n_bxs - 1)
  {
    if(mask[row + col + 1] == 0)
    {
      row += 1;
    }
    else
    {
      break;
    }
  }
  col += row + 1;
  row = 0;
}

}

void nms_cuda(float* boxes, int* elims, int* mask, float thresh,
            int n_boxes, cudaStream_t stream)
{
cudaError_t err;
int blocks_per_dim = n_boxes / 12 + 1;
dim3 blocks(blocks_per_dim, blocks_per_dim);
dim3 threadsPerBlock(12, 12);
nms_kern<<<blocks, threadsPerBlock, 0, stream>>>(boxes, elims, mask,
                                                 thresh, n_boxes);
err = cudaGetLastError();
if (err != cudaSuccess)
{
  fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
  exit(-1);
}
}

}
