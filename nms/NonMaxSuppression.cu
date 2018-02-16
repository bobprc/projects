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
  // elims[elim_idx(x, y, n)] = false if box x eliminates box y,
  // true otherwise. n is the total number of boxes.
  return x <= (n-1)/2? x*(n-1) + y: n*(n-2-x) + y+1;
}

template <typename T>
__global__ void nms_forward_elims_kernel(
  const T *boxes, 
  bool *elims,
  float thresh,
  int num_boxes)
{
  // For each pair of boxes, determine whether the higher-scoring box
  // eliminates the lower-scoring one. Store this information in the array
  // "elims". Each thread handles a pair of boxes.

  // Use a flat index so that we have only batch_size underutilised blocks.
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int batch_shift = gridDim.x * blockDim.x * blockIdx.y;

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
    box_y_idx = idx - midpt + box_x_idx + 1 num_boxes * blockIdx.y;
  }
  else
  {
    box_x_idx = base + batch_shift + num_boxes * blockIdx.y;
    box_y_idx = idx - midpt + num_boxes + num_boxes * blockIdx.y;
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

  // Write false if box x eliminates box y.
  elims[idx + batch_shift] = (iou <= thresh);

}

__global__ void nms_forward_mask_kernel(Tensor *mask, bool* elims, int num_boxes)
{
  // Given "elims", calculate the mask. Unfortunately this
  // requires global synchronisation, so launch only one block
  // per batch element and use a for loop to cover all the boxes.
  int col = 0;
  int batch_shift = (num_boxes*(num_boxes-1))/2*blockIdx.y;
  while(col < num_boxes-1)
  {
    for(int i = threadIdx.x; i < num_boxes-1; i+=blockDim.x)
      if(i >= col)
        mask[i+1+blockIdx.y*num_boxes] = (
           mask[i+1+blockIdx.y*num_boxes] ?
           elims[elim_idx(col, i, num_boxes)+batch_shift] : false);
    __syncthreads();
    ++col;
    while((col < num_boxes - 1) && (mask[col+blockIdx.y] == 0))
      ++col;
  }
}

Tensor NonMaxSupression_forward_cuda(const Tensor& input, const Tensor& scores, float thresh)
{

  AT_ASSERT(input.ndimension() == 3, "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_ASSERT(scores.ndimension() == 4, "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_ASSERT(input.size(0) == scores.size(0), "First and second arguments must have equal-sized first dimension");
  AT_ASSERT(input.size(1) == scores.size(1), "First and second arguments must have equal-sized second dimension");
  AT_ASSERT(input.size(2) == 4, "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");


  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});
  int n_pairs = (num_boxes*(num_boxes-1))/2;
  
  bool *elims;
  cudaMalloc(elims, n_pairs*batch_size*sizeof(bool));
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "Failed to allocate memory for NonMaxSuppression");

  dim3 block(512);
  dim3 grid((n_paris-1+512)/512, batch_size);
  calc_elims<<<grid, block, 0, globalContext().getCurrentCUDAStream()>>>(
                                                 boxes.data<float>(),
                                                 elims, 
                                                 thresh,
                                                 num_boxes);
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "nms_forward_elims_kernel failed");
  dim3 block(512);
  dim3 grid(batch_size);
  calc_mask<<<grid, block, 0, globalContext().getCurrentCUDAStream()>>>(
                                    *elims, 
                                    mask.data<bool>,
                                    num_boxes);
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "nms_forward_mask_kernel failed");

  return mask;
}
