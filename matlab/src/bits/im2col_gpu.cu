/** @file im2col.cu
 ** @brief Image to columns and back (GPU)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2col.hpp"
#include "gpu.hpp"

/* ---------------------------------------------------------------- */
/*                                                     im2col (GPU) */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void
im2col_gpu_kernel(T* stacked,
                  T const* data,
                  const int numPatchesX,
                  const int numPatchesY,
                  const int numPatchSlices,
                  const int width,
                  const int height,
                  const int windowWidth,
                  const int windowHeight,
                  const int strideX,
                  const int strideY,
                  const int padLeft,
                  const int padTop)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    /* 
      get the patch slice (x,y,z) to copy
     */
    int x = index ;
    int y = x / numPatchesX ;
    int z = y / numPatchesY ;
    x %= numPatchesX ;
    y %= numPatchesY ;

    /* 
     pick the top-left corer of the patch slice in the input image
     */
    int x_data = x * strideX - padLeft ;
    int y_data = y * strideY - padTop ;
    data += (z * height + y_data) * width + x_data ;

    /* 
     pick the column of the stacked image which contains this patch,
     and move down along the column at the beginning of the patch slice
     */
    int patchSliceOffset = (windowWidth*windowHeight) * z ;
    stacked += (numPatchesY * patchSliceOffset + y) * numPatchesX + x ;

    /*
     copy the patch slice
     */
    for (int v = 0 ; v < windowHeight ; ++v) {
      for (int u = 0 ; u < windowWidth ; ++u) {
        if (y_data + v >= 0 &&
            y_data + v < height &&
            x_data + u >= 0 &&
            x_data + u < width) {
          *stacked = data[v * width + u] ;
        } else {
          *stacked = 0 ;
        }
        stacked += (numPatchesX*numPatchesY) ;
      }
    }
  }
}

template <typename T>
void im2col_gpu(T* stacked,
                T const* data,
                size_t width,
                size_t height,
                size_t depth,
                size_t windowWidth,
                size_t windowHeight,
                size_t strideX,
                size_t strideY,
                size_t padLeft,
                size_t padRight,
                size_t padTop,
                size_t padBottom)
{
  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numPatchSlices = numPatchesX * numPatchesY * depth ;

  /*
   Each kernel copies a feature dimension of a patch.
   */
  im2col_gpu_kernel<T>
  <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (stacked,
   data,
   numPatchesX,
   numPatchesY,
   numPatchSlices,
   width, height,
   windowWidth, windowHeight,
   strideX, strideY,
   padLeft, padTop) ;

  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout
    <<"im2col: CUDA kernel error ("
    <<cudaGetErrorString(cudaPeekAtLastError())
    <<")"<<std::endl ;
  }
}

// Explicit instantiation
template void im2col_gpu<float>(float* stacked,
                                float const* data,
                                size_t width,
                                size_t height,
                                size_t depth,
                                size_t windowWidth,
                                size_t windowHeight,
                                size_t strideX,
                                size_t strideY,
                                size_t padLeft,
                                size_t padRight,
                                size_t padTop,
                                size_t padBottom);

template void im2col_gpu<double>(double* stacked,
                                 double const* data,
                                 size_t width,
                                 size_t height,
                                 size_t depth,
                                 size_t windowWidth,
                                 size_t windowHeight,
                                 size_t strideX,
                                 size_t strideY,
                                 size_t padLeft,
                                 size_t padRight,
                                 size_t padTop,
                                 size_t padBottom);

/* ---------------------------------------------------------------- */
/*                                        im2col with indices (GPU) */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void
im2col_gpu_indexed_size_1_kernel(T* __restrict__ stacked,
                                 T const* __restrict__ data,
                                 int const* __restrict__ indices,
                                 const int indicesLength,
                                 const int numPatchSlices,
                                 const int dataSize)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    int x = index;
    int z = index / indicesLength;
    x %= indicesLength;

    int idxValue = indices[x];
    stacked[index] = (idxValue != -1) ? data[z * dataSize + idxValue] : 0;
  }
}

template <typename T>
__global__ void
im2col_gpu_indexed_kernel(T* __restrict__ stacked,
                          T const* __restrict__ data,
                          int const* __restrict__ indices,
                          const int maskIndicesLength,
                          const int dataSize,
                          const int depth,
                          const int depthCol,
                          const int size,
                          const int numPatchSlices)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    int x = index;
    int s = x / maskIndicesLength;
    int d = s / size;
    int c = d / depthCol;
    x %= maskIndicesLength;
    s %= size;
    d %= depthCol;

    int idxValue = indices[d * maskIndicesLength + x];
    stacked[index] = (idxValue != -1) ? data[(s * depth + c) * dataSize + idxValue] : 0;
  }
}

template <typename T>
void im2col_indexed_gpu(T* stacked,
                        T const* data,
                        int const* indices,
                        int indicesLength,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t size,
                        size_t windowWidth,
                        size_t windowHeight)
{
  int numPatchSlices = indicesLength * depth * size ;
  int depthCol = windowWidth * windowHeight;
  int maskIndicesLength = indicesLength / depthCol;

  if (size == 1) {
    im2col_gpu_indexed_size_1_kernel<T>
    <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (stacked,
     data,
     indices, indicesLength,
     numPatchSlices,
     width * height) ; 
  } else {
    im2col_gpu_indexed_kernel<T>
    <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (stacked,
     data,
     indices,
     maskIndicesLength,
     width * height,
     depth,
     depthCol,
     size,
     numPatchSlices) ;
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout
    <<"im2col_indexed_gpu: CUDA kernel error ("
    <<cudaGetErrorString(cudaPeekAtLastError())
    <<")"<<std::endl ;
  }
}

// Explicit instantiation
template void im2col_indexed_gpu<float>(float* stacked,
                                        float const* data,
                                        int const* indices,
                                        int indicesLength,
                                        size_t width,
                                        size_t height,
                                        size_t depth,
                                        size_t size,
                                        size_t windowWidth,
                                        size_t windowHeight);

template void im2col_indexed_gpu<double>(double* stacked,
                                         double const* data,
                                         int const* indices,
                                         int indicesLength,
                                         size_t width,
                                         size_t height,
                                         size_t depth,
                                         size_t size,
                                         size_t windowWidth,
                                         size_t windowHeight);

/* ---------------------------------------------------------------- */
/*                                                     col2im (GPU) */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void col2im_gpu_kernel(T* data,
                                  T const* stacked,
                                  const int numPatchesX,
                                  const int numPatchesY,
                                  const int dataVolume,
                                  const int width,
                                  const int height,
                                  const int depth,
                                  const int windowWidth,
                                  const int windowHeight,
                                  const int strideX,
                                  const int strideY,
                                  const int padLeft,
                                  const int padTop)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume)
  {
    T accumulator = 0 ;
    /*
     This kernel accumulates on data[index] all elements in stacked
     that receive copies of data[index] in im2col.
     
     Consider coordinate (x_data,y_data) in the input image. Relative to patch
     (x,y), this has offset
     
     u = x_data - (x * strideX - padLeft)
     v = y_data - (y * strideY - padRight)
     
     In particular, (x_data,y_data) is contained (and hence contributes)
     to patch (x,y) if, and only if,
     
     0 <= u < windowWidth  <==>  1) x_data >= x * strideX - padLeft
                                 2) x_data <  x * strideX - padLeft + windowWidth
     
     and similar for y.
     
     Hence, the patches that contribute to (x_data,y_data) are given
     by indexes (x,y) such that
     
     (x_data + padLeft - windowWidth)/stride < x
         <= (x_data + padLeft)/stride
     
     or, accounting for the boundaries,

       x1 <= x <= x2, such that
         x1 = max(0,  1 + floor(x_data + padLeft - windowWidth)/stride),
         x2 = min(numPatchesX-1,  floor(x_data + padLeft)/stride),
     
     and similar for y.
     
     Note that (x_data + padLeft - windowWidth) may be negative. In this case,
     the C convention for rounding division towards zero fails to compute
     the floor() properly. Instead, we check this case explicitly and set
     */

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - windowWidth ;
    int dy = y_data + padTop - windowHeight ;
    int x1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int y1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int x2 = min((x_data + padLeft) / strideX, numPatchesX - 1) ;
    int y2 = min((y_data + padTop) / strideY, numPatchesY - 1) ;

    /*
     Knowing which patches (x,y) contribute to (x_data,y_data) is not enough;
     we need to determine the specific element within each patch. This
     is given by the offset as given above:
     
     u(x) = x_data - (x * strideX - padLeft)
     v(y) = y_data - (y * strideY - padRight)
     
     Now we can comptute the indeces of the elements of stacked[] to accumulate:
     
     stackedIndex(x,y) = 
         (y * numPatchesX + x) +                 // column offset
         ((z * windowHeight + v(y)) * windowWidth + u(x)) *  // within patch offset
            (numPatchesX*numPatchesY)

     Substituting the expression fo u(x), we find

     stackedIndex(x,y) =
         = (y * numPatchesX + x)
         + ((z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
           * (numPatchesX*numPatchesY)
         - ((y * strideY) * windowWidth + x * strideX)
           * (numPatchesX*numPatchesY)
         = (z * windowHeight + y_data + padTop) * windowWidth + x_data + padLeft)
         + x * (1 - strideX*numPatchesY*numPatchesX)
         + y * (1 - strideY*numPatchesY*windowWidth)*numPatchesX ;

     */

    int deltax = (1 - strideX * numPatchesY * numPatchesX) ;
    int deltay = (1 - strideY * numPatchesY * windowWidth) * numPatchesX ;
    stacked += ((z * windowHeight + y_data + padTop) * windowWidth + (x_data + padLeft)) * (numPatchesX*numPatchesY) ;

    for (int y = y1 ; y <= y2 ; ++ y) {
      for (int x = x1 ; x <= x2 ; ++ x) {
        accumulator += stacked[y * deltay + x * deltax];
      }
    }
    data[index] = accumulator;
  }
}

template <typename T>
void col2im_gpu(T* data,
                T const* stacked,
                size_t width,
                size_t height,
                size_t depth,
                size_t windowWidth,
                size_t windowHeight,
                size_t strideX,
                size_t strideY,
                size_t padLeft,
                size_t padRight,
                size_t padTop,
                size_t padBottom)
{
  /*
   each kernel integrates all contributions to a particular element
   of data.
   */
  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int dataVolume = width * height * depth ;

  col2im_gpu_kernel<T>
  <<< divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (data,
   stacked,
   numPatchesX,
   numPatchesY,
   dataVolume,
   width, height, depth,
   windowWidth, windowHeight,
   strideX, strideY,
   padLeft, padTop) ;

  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout
    <<"col2im: CUDA kernel error ("
    <<cudaGetErrorString(cudaPeekAtLastError())
    <<")"<<std::endl ;
  }
}

template void col2im_gpu<float>(float* data,
                                float const* stacked,
                                size_t width,
                                size_t height,
                                size_t depth,
                                size_t windowWidth,
                                size_t windowHeight,
                                size_t strideX,
                                size_t strideY,
                                size_t padLeft,
                                size_t padRight,
                                size_t padTop,
                                size_t padBottom);

template void col2im_gpu<double>(double* data,
                                 double const* stacked,
                                 size_t width,
                                 size_t height,
                                 size_t depth,
                                 size_t windowWidth,
                                 size_t windowHeight,
                                 size_t strideX,
                                 size_t strideY,
                                 size_t padLeft,
                                 size_t padRight,
                                 size_t padTop,
                                 size_t padBottom);

/* ---------------------------------------------------------------- */
/*                                        col2im with indices (GPU) */
/* ---------------------------------------------------------------- */

template <typename T>
__global__ void
col2im_gpu_indexed_size_1_kernel(T* __restrict__ data,
                                 T const* __restrict__ stacked,
                                 int const* __restrict__ indices,
                                 const int indicesLength,
                                 const int numPatchSlices,
                                 const int width,
                                 const int height,
                                 const int depth)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    int idx = index % indicesLength;
    int z = index / indicesLength;

    int idxValue = indices[idx];
    if (idxValue != -1) {
      atomicAdd(data + z * width * height + idxValue, stacked[index]);
    }
  }
}

template <typename T>
__global__ void
col2im_gpu_indexed_kernel(T* __restrict__ data,
                          T const* __restrict__ stacked,
                          int const* __restrict__ indices,
                          const int maskIndicesLength,
                          const int dataSize,
                          const int depth,
                          const int depthCol,
                          const int size,
                          const int numPatchSlices)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    int x = index;
    int s = x / maskIndicesLength;
    int d = s / size;
    int c = d / depthCol;
    x %= maskIndicesLength;
    s %= size;
    d %= depthCol;

    int idxValue = indices[d * maskIndicesLength + x];
    if (idxValue != -1) {
      atomicAdd(data + (s * depth + c) * dataSize + idxValue, stacked[index]) ;
    }
  }
}

template <typename T>
void col2im_indexed_gpu(T* data,
                        T const* stacked,
                        int const* indices,
                        int indicesLength,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t size,
                        size_t windowWidth,
                        size_t windowHeight)
{
  int numPatchSlices = indicesLength * depth * size ;
  int depthCol = windowWidth * windowHeight;
  int maskIndicesLength = indicesLength / depthCol;

  cudaMemset(data, 0, sizeof(T)*width*height*depth*size);

  if (size == 1) {
    col2im_gpu_indexed_size_1_kernel<T>
    <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (data,
     stacked,
     indices, indicesLength,
     numPatchSlices,
     width, height, depth) ;
  } else {
    col2im_gpu_indexed_kernel<T>
    <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
    (data,
     stacked,
     indices,
     maskIndicesLength,
     width * height,
     depth,
     depthCol,
     size,
     numPatchSlices) ;
  }

  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout
    <<"col2im_indexed_gpu: CUDA kernel error ("
    <<cudaGetErrorString(cudaPeekAtLastError())
    <<")"<<std::endl ;
  }
}

// Explicit instantiation
template void col2im_indexed_gpu<float>(float* data,
                                        float const* stacked,
                                        int const* indices,
                                        int indicesLength,
                                        size_t width,
                                        size_t height,
                                        size_t depth,
                                        size_t size,
                                        size_t windowWidth,
                                        size_t windowHeight);

template <typename T>
__global__ void transpose23_kernel(T* transposed,
                                   const T* data,
                                   const int d1,
                                   const int d2,
                                   const int d3,
                                   const int numPatchSlices)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    int x = index;
    int y = x / d1;
    int z = y / d2;
    x %= d1;
    y %= d2;

    transposed[y*(d1*d3) + z*d1 + x] = data[z*(d1*d2) + y*d1 + x];
  }
}

template <typename T>
void transpose23_gpu(T* transposed,
                     T const* data,
                     size_t d1,
                     size_t d2,
                     size_t d3)
{
  int numPatchSlices = d1 * d2 * d3 ;

  /*
   Each kernel copies a feature dimension of a patch.
   */
  transpose23_kernel<T>
  <<< divideUpwards(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (transposed,
   data,
   d1,
   d2,
   d3,
   numPatchSlices) ;

  if (cudaPeekAtLastError() != cudaSuccess) {
    std::cout
    <<"transpose23_gpu: CUDA kernel error ("
    <<cudaGetErrorString(cudaPeekAtLastError())
    <<")"<<std::endl ;
  }
}

template void transpose23_gpu<float>(float* transposed,
                                     float const* data,
                                     size_t d1,
                                     size_t d2,
                                     size_t d3);

template void transpose23_gpu<double>(double* transposed,
                                      double const* data,
                                      size_t d1,
                                      size_t d2,
                                      size_t d3);
