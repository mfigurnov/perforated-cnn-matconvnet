/** @file im2col.cpp
 ** @brief Image to columns and back (CPU)
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2col.hpp"
#include <string.h>

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a-b+1)/b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

static inline int static_max(int a, int b) {
  return (a>=b) ? a:b ;
}

static inline int static_min(int a, int b) {
  return (a<=b) ? a:b ;
}

/* ---------------------------------------------------------------- */
/*                                                     im2col (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void im2col_cpu(T* stacked,
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
  int numRows = windowWidth * windowHeight * depth ;

  /* 
   Fill a row of the stacked image at a time. Since patches are stored
   along the columns, scanning a row menas visiting all patche once.
   Each row corresponds to a particular offset within each patch.
   
   In this manner, as we fill a row
   we tend to access spatially adiacent elements
   in the input image, particulary for small strides.
   */
  for (int row = 0; row < numRows ; ++row) {
    /* 
     Get the patch offset corresponding to this row of the stacked
     image.
     */
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;
    u %= windowWidth ;
    v %= windowHeight ;

    /*
     Filling this row amounts to visiting all the pixels in the input
     image that appear at a given offset in the outut patches. Accounting
     for the subsampling of the output patches and input padding,
     these pixels are given by
     
     x_data(x) = x * strideX + u - padLeft,  0 <= x < numPatchesX
     y_data(y) = y * strideY + v - padTop,   0 <= y < numPatchesY
     z_data(z) = z.
     
     Here (x,y) are the spatial indexes of the output patches. Depedning
     on the padding, some of these values will read pixels outised
     the input image, which should default to 0. In particular this happens
     if
     
     x_data(x) < 0 <=> x < (padLeft - u) / stride 
                   <=> x < ceil((padLeft - u) / stride)
     x_data(x) >= width <=> x >= (width + padLeft - u) / stride
                        <=> x >= ceil((width + padLeft - u) / stride)
     
     and the same for y.
     */

    int x0 = static_max(0, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_max(0, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX,  ceil_divide(width + padLeft - u, strideX)) ;
    int y1 = static_min(numPatchesY, ceil_divide(height + padTop - v, strideY)) ;

    for (int y = 0 ; y < y0 ; ++y) {
      for (int x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
    for (int y = y0 ; y < y1 ; ++y) {
      int y_data = y * strideY + v - padTop ;
      int x_data = x0 * strideX + u - padLeft ;
      T const * b = data + (z * height + y_data) * width + x_data ;

      for (int x = 0 ; x < x0 ; ++x) {
        *stacked++ = 0 ;
      }
      for (int x = x0 ; x < x1 ; ++x) {
        *stacked++ = *b ;
        b += strideX ;
      }
      for (int x = x1 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
    for (int y = y1 ; y < numPatchesY ; ++y) {
      for (int x = 0 ; x < numPatchesX ; ++x) {
        *stacked++ = 0 ;
      }
    }
  }
}

template void im2col_cpu<float>(float* stacked,
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

template void im2col_cpu<double>(double* stacked,
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
/*                          im2col with precalculated indices (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void im2col_indexed_cpu(T* __restrict__ stacked,
                        T const* __restrict__ data,
                        int const* __restrict__ indices,
                        int indicesSize,
                        int width,
                        int height,
                        int depth,
                        int size,
                        int windowWidth,
                        int windowHeight)
{
  int depthCol = windowWidth * windowHeight;
  int maskIndicesLength = indicesSize / depthCol;

  for (int s = 0; s < size; ++s) {
    for (int c = 0; c < depth; ++c) {
      for (int d = 0; d < depthCol; ++d) {
        for (int x = 0; x < maskIndicesLength; ++x) {
          int idxValue = indices[d * maskIndicesLength + x];
          stacked[((c * depthCol + d) * size + s) * maskIndicesLength + x] =
            (idxValue != -1) ? data[(s * depth + c) * width * height + idxValue] : 0;
        }
      }
    }
  }
}

template void im2col_indexed_cpu<float>(float* stacked,
                                        float const* data,
                                        int const* indices,
                                        int indicesSize,
                                        int width,
                                        int height,
                                        int depth,
                                        int size,
                                        int windowWidth,
                                        int windowHeight);

/* ---------------------------------------------------------------- */
/*                                                     col2im (CPU) */
/* ---------------------------------------------------------------- */

template <typename T>
void col2im_cpu(T* data,
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
  int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  int numRows = windowWidth * windowHeight * depth ;

  memset(data, 0, sizeof(T) * width * height * depth) ;

  /*
   Do the converse of im2col, still scanning rows of the stacked image.
   See comments of im2col for an explanation of the algorithms.
   */
  for (int row = 0; row < numRows ; ++row) {
    int u = row ;
    int v = u / windowWidth ;
    int z = v / windowHeight ;
    u %= windowWidth ;
    v %= windowHeight ;

    int x0 = static_max(0, ceil_divide(padLeft - u, strideX)) ;
    int y0 = static_max(0, ceil_divide(padTop - v, strideY)) ;
    int x1 = static_min(numPatchesX, ceil_divide(width + padLeft - u, strideX)) ;
    int y1 = static_min(numPatchesY, ceil_divide(height + padTop - v, strideY)) ;

    stacked += numPatchesX * y0 ;
    for (int y = y0 ; y < y1 ; ++y) {
      int y_data = y * strideY + v - padTop ;
      int x_data = x0 * strideX + u - padLeft ;
      T * b = data + (z * height + y_data) * width + x_data ;
      stacked += x0 ;
      for (int x = x0 ; x < x1 ; ++x) {
        *b += *stacked++ ;
        b += strideX ;
      }
      stacked += numPatchesX - x1 ;
    }
    stacked += numPatchesX * (numPatchesY - y1) ;
  }
}

template void col2im_cpu<float>(float* data,
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

template void col2im_cpu<double>(double* data,
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

template<typename T>
void col2im_indexed_cpu(T* data,
                        T const* stacked,
                        int const* indices,
                        int indicesSize,
                        int width,
                        int height,
                        int depth,
                        int size,
                        int windowWidth,
                        int windowHeight)
{
  memset(data, 0, sizeof(T)*width*height*depth*size);

  int depthCol = windowWidth * windowHeight;
  int maskIndicesLength = indicesSize / depthCol;

  for (int s = 0; s < size; ++s) {
    for (int c = 0; c < depth; ++c) {
      for (int d = 0; d < depthCol; ++d) {
        for (int x = 0; x < maskIndicesLength; ++x) {
          int idxValue = indices[d * maskIndicesLength + x];
          if (idxValue != -1) {
            data[(s * depth + c) * width * height + idxValue] += stacked[((c * depthCol + d) * size + s) * maskIndicesLength + x];
          }
        }
      }
    }
  }
}

template void col2im_indexed_cpu<float>(float* data,
                                        float const* stacked,
                                        int const* indices,
                                        int indicesSize,
                                        int width,
                                        int height,
                                        int depth,
                                        int size,
                                        int windowWidth,
                                        int windowHeight);

template void col2im_indexed_cpu<double>(double* data,
                                         double const* stacked,
                                         int const* indices,
                                         int indicesSize,
                                         int width,
                                         int height,
                                         int depth,
                                         int size,
                                         int windowWidth,
                                         int windowHeight);


template<typename T>
void transpose23_cpu(T* transposed,
                     T const* data,
                     size_t d1,
                     size_t d2,
                     size_t d3)
{
  for (int k = 0; k < d3; ++k) {
    for (int j = 0; j < d2; ++j) {
      memcpy(transposed + j*(d1*d3) + k*d1, data + k*(d1*d2) + j*d1, d1 * sizeof(float));
    }
  }
}
template void transpose23_cpu<float>(float* transposed,
                                     float const* data,
                                     size_t d1,
                                     size_t d2,
                                     size_t d3);

template void transpose23_cpu<double>(double* transposed,
                                      double const* data,
                                      size_t d1,
                                      size_t d2,
                                      size_t d3);


void conv_indices_cpu(int* indices,
                      int indicesLength,
                      int const* inIndices,
                      int const* maskIndices,
                      int maskIndicesLength,
                      int width,
                      int height,
                      int depth,
                      int windowWidth,
                      int windowHeight,
                      int strideX,
                      int strideY,
                      int padLeft,
                      int padRight,
                      int padTop,
                      int padBottom)
{
  int height_col = (height + (padTop + padBottom) - windowHeight) / strideY + 1;
  int width_col = (width + (padLeft + padRight) - windowWidth) / strideX + 1;
  int depth_col = windowHeight * windowWidth;
  assert(indicesLength == maskIndicesLength * depth_col);
  for (int c = 0; c < depth_col; ++c) {
    int w_offset = c % windowWidth;
    int h_offset = (c / windowWidth) % windowHeight;
    for (int i = 0; i < maskIndicesLength; ++i) {
      int index = maskIndices ? maskIndices[i] : i;
      int h = index / width_col;
      int w = index % width_col;
      int h_pad = h * strideY - padTop + h_offset;
      int w_pad = w * strideX - padLeft + w_offset;
      if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
        int curIndex = h_pad * width + w_pad;
        if (inIndices) {
          curIndex = inIndices[curIndex];
        }
        indices[c * maskIndicesLength + i] = curIndex;
      }
      else
        indices[c * maskIndicesLength + i] = -1;
    }
  }
}
