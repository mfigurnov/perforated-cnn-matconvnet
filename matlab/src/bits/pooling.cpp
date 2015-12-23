/** @file pooling.cpp
 ** @brief Max pooling filters (CPU)
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "pooling.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

#include <cassert>
#include <cstring>
#include <cmath>


/* ---------------------------------------------------------------- */
/*                                                 maxPooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename T>
void pooling_cpu(T* pooled,
                 T const* data,
                 PoolMethod method,
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
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  switch (method) {
    case NN_POOL_MAX :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = x * (signed)strideX - (signed)padLeft ;
            int y1 = y * (signed)strideY - (signed)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
            T bestValue = data[y1 * width + x1] ;
            for (int v = y1 ; v < y2 ; ++v) {
              for (int u = x1 ; u < x2 ; ++u) {
                bestValue = std::max(bestValue, data[v * width + u]) ;
              }
            }
            pooled[y * pooledWidth + x] = bestValue ;
          }
        }
        data += width*height ;
        pooled += pooledWidth*pooledHeight ;
      }
      break;
    case NN_POOL_AVG :
      for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < pooledHeight; ++y) {
          for (int x = 0; x < pooledWidth; ++x) {
            int x1 = x * (signed)strideX - (signed)padLeft ;
            int y1 = y * (signed)strideY - (signed)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
            T accum = 0 ;
            T poolSize = (y2 - y1) * (x2 - x1);
            for (int v = y1 ; v < y2 ; ++v) {
              for (int u = x1 ; u < x2 ; ++u) {
                accum += data[v * width + u] ;
              }
            }
            pooled[y * pooledWidth + x] = accum / poolSize ;
          }
        }
        data += width*height ;
        pooled += pooledWidth*pooledHeight ;
      }
      break;
    default:
      assert(false) ;
  }


}

template
void pooling_cpu<float>(float* pooled,
                        float const* data,
                        PoolMethod method,
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
                        size_t padBottom) ;

template
void pooling_cpu<double>(double* pooled,
                         double const* data,
                         PoolMethod method,
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
                         size_t padBottom) ;


/* ---------------------------------------------------------------- */
/*                                         maxPoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

/* 
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */
template<typename T>
void poolingBackward_cpu(T* dzdx,
                         T const* data,
                         T const* dzdy,
                         PoolMethod method,
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
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  switch (method) {
    case NN_POOL_MAX :
      for (int z = 0; z < depth; ++z) {
        for (int py = 0; py < pooledHeight; ++py) {
          for (int px = 0; px < pooledWidth; ++px) {
            int x1 = px * (int)strideX - (int)padLeft ;
            int y1 = py * (int)strideY - (int)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
            int bestIndex = y1 * width + x1 ;
            T bestValue = data[bestIndex] ;
            for (int y = y1 ; y < y2 ; ++y) {
              for (int x = x1 ; x < x2 ; ++x) {
                int index = y * width + x ;
                T value = data[index] ;
                if (value > bestValue) {
                  bestValue = value ;
                  bestIndex = index ;
                }
              }
            }
            dzdx[bestIndex] += dzdy[py * pooledWidth + px] ;
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
      break;
    case NN_POOL_AVG :
      for (int z = 0; z < depth; ++z) {
        for (int py = 0; py < pooledHeight; ++py) {
          for (int px = 0; px < pooledWidth; ++px) {
            int x1 = px * (int)strideX - (int)padLeft ;
            int y1 = py * (int)strideY - (int)padTop ;
            int x2 = std::min(x1 + windowWidth, width) ;
            int y2 = std::min(y1 + windowHeight, height) ;
            x1 = std::max(x1, 0) ;
            y1 = std::max(y1, 0) ;
            T poolSize = (y2 - y1) * (x2 - x1);
            for (int y = y1 ; y < y2 ; ++y) {
              for (int x = x1 ; x < x2 ; ++x) {
                dzdx[y * width + x] += dzdy[py * pooledWidth + px] / poolSize;
              }
            }
          }
        }
        data += width*height ;
        dzdx += width*height ;
        dzdy += pooledWidth*pooledHeight ;
      }
     break;
    default:
      assert(false) ;
  }
}

template
void poolingBackward_cpu<float>(float* dzdx,
                                float const* data,
                                float const* dzdy,
                                PoolMethod method,
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
                                size_t padBottom) ;

template
void poolingBackward_cpu<double>(double* dzdx,
                                 double const* data,
                                 double const* dzdy,
                                 PoolMethod method,
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
                                 size_t padBottom) ;


template<typename T, int windowSize>
void max_pooling_cpu_fast_internal(T* __restrict__ pooled,
                                   T const* __restrict__ data,
                                   int const* __restrict__ indices,
                                   size_t dataSize,
                                   size_t depth,
                                   size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T bestValue;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        T value = data[index];
        if (u == 0 || value > bestValue) {
          bestValue = value;
        }
      }
      pooled[x] = bestValue;
    }
    data += dataSize;
    pooled += pooledSize;
  }
}

template<typename T>
void max_pooling_cpu_fast_internal_2(T* __restrict__ pooled,
                                     T const* __restrict__ data,
                                     int const* __restrict__ indices,
                                     size_t dataSize,
                                     size_t depth,
                                     size_t windowSize,
                                     size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T bestValue;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        T value = data[index];
        if (u == 0 || value > bestValue) {
          bestValue = value;
        }
      }
      pooled[x] = bestValue;
    }
    data += dataSize;
    pooled += pooledSize;
  }
}

template<typename T, int windowSize>
void avg_pooling_cpu_fast_internal(T* __restrict__ pooled,
                                   T const* __restrict__ data,
                                   int const* __restrict__ indices,
                                   size_t dataSize,
                                   size_t depth,
                                   size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T accum = 0;
      T poolSize = 0;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        if (index != -1) {
          accum += data[index];
          ++poolSize;
        }
      }
      pooled[x] = accum / poolSize;
    }
    data += dataSize;
    pooled += pooledSize;
  }
}

template<typename T>
void avg_pooling_cpu_fast_internal_2(T* __restrict__ pooled,
                                     T const* __restrict__ data,
                                     int const* __restrict__ indices,
                                     size_t dataSize,
                                     size_t depth,
                                     size_t windowSize,
                                     size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T accum = 0;
      T poolSize = 0;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        if (index != -1) {
          accum += data[index];
          ++poolSize;
        }
      }
      pooled[x] = accum / poolSize;
    }
    data += dataSize;
    pooled += pooledSize;
  }
}

template<typename T>
void pooling_cpu_fast(T* pooled,
                      T const* data,
                      int const* indices,
                      PoolMethod method,
                      size_t dataSize,
                      size_t depth,
                      size_t windowSize,
                      size_t pooledSize)
{
#define MAX_POOL_CPU(_windowSize) case _windowSize: \
  max_pooling_cpu_fast_internal<T, _windowSize>\
    (pooled, data, indices, dataSize, depth, pooledSize); break
#define AVG_POOL_CPU(_windowSize) case _windowSize: \
  avg_pooling_cpu_fast_internal<T, _windowSize>\
    (pooled, data, indices, dataSize, depth, pooledSize); break

  switch (method) {
    case NN_POOL_MAX :
      switch (windowSize) {
        MAX_POOL_CPU(1);
        MAX_POOL_CPU(4);
        MAX_POOL_CPU(9);
        MAX_POOL_CPU(16);
        MAX_POOL_CPU(25);
        MAX_POOL_CPU(36);
        MAX_POOL_CPU(49);
        default:
          max_pooling_cpu_fast_internal_2<T>
            (pooled, data, indices, dataSize, depth, windowSize, pooledSize);
          break;
      }
      break;
    case NN_POOL_AVG:
      switch (windowSize) {
        AVG_POOL_CPU(1);
        AVG_POOL_CPU(4);
        AVG_POOL_CPU(9);
        AVG_POOL_CPU(16);
        AVG_POOL_CPU(25);
        AVG_POOL_CPU(36);
        AVG_POOL_CPU(49);
        default:
          avg_pooling_cpu_fast_internal_2<T>
            (pooled, data, indices, dataSize, depth, windowSize, pooledSize);
          break;
      }
      break;
    default:
      assert(false);
  }
#undef MAX_POOL_CPU
#undef AVG_POOL_CPU
}

template
void pooling_cpu_fast<float>(float* pooled,
                             float const* data,
                             int const* indices,
                             PoolMethod method,
                             size_t dataSize,
                             size_t depth,
                             size_t windowSize,
                             size_t pooledSize);

template
void pooling_cpu_fast<double>(double* pooled,
                              double const* data,
                              int const* indices,
                              PoolMethod method,
                              size_t dataSize,
                              size_t depth,
                              size_t windowSize,
                              size_t pooledSize);

template<typename T, int windowSize>
void max_pooling_backward_cpu_fast_internal(T* __restrict__ dzdx,
                                            T const* __restrict__ data,
                                            T const* __restrict__ dzdy,
                                            int const* __restrict__ indices,
                                            size_t dataSize,
                                            size_t depth,
                                            size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T bestValue;
      int bestIndex;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        T value = data[index];
        if (u == 0 || value > bestValue) {
          bestIndex = index;
          bestValue = value;
        }
      }
      dzdx[bestIndex] += dzdy[x];
    }
    data += dataSize;
    dzdx += dataSize;
    dzdy += pooledSize;
  }
}

template<typename T>
void max_pooling_backward_cpu_fast_internal_2(T* __restrict__ dzdx,
                                              T const* __restrict__ data,
                                              T const* __restrict__ dzdy,
                                              int const* __restrict__ indices,
                                              size_t dataSize,
                                              size_t depth,
                                              size_t windowSize,
                                              size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T bestValue;
      int bestIndex;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        T value = data[index];
        if (u == 0 || value > bestValue) {
          bestIndex = index;
          bestValue = value;
        }
      }
      dzdx[bestIndex] += dzdy[x];
    }
    data += dataSize;
    dzdx += dataSize;
    dzdy += pooledSize;
  }
}

template<typename T, int windowSize>
void avg_pooling_backward_cpu_fast_internal(T* __restrict__ dzdx,
                                            T const* __restrict__ data,
                                            T const* __restrict__ dzdy,
                                            int const* __restrict__ indices,
                                            size_t dataSize,
                                            size_t depth,
                                            size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T poolSize = 0;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        if (index != -1) {
          ++poolSize;
        }
      }

      if (poolSize) {
        #pragma unroll
        for (int u = 0; u < windowSize; ++u) {
          int index = indices[x * windowSize + u];
          if (index != -1) {
            dzdx[index] += dzdy[x] / poolSize;
          }
        }
      }
    }
    data += dataSize;
    dzdx += dataSize;
    dzdy += pooledSize;
  }
}

template<typename T>
void avg_pooling_backward_cpu_fast_internal_2(T* __restrict__ dzdx,
                                              T const* __restrict__ data,
                                              T const* __restrict__ dzdy,
                                              int const* __restrict__ indices,
                                              size_t dataSize,
                                              size_t depth,
                                              size_t windowSize,
                                              size_t pooledSize)
{
  for (int z = 0; z < depth; ++z) {
    for (int x = 0; x < pooledSize; ++x) {
      T poolSize = 0;
      #pragma unroll
      for (int u = 0; u < windowSize; ++u) {
        int index = indices[x * windowSize + u];
        if (index != -1) {
          ++poolSize;
        }
      }

      if (poolSize) {
        #pragma unroll
        for (int u = 0; u < windowSize; ++u) {
          int index = indices[x * windowSize + u];
          if (index != -1) {
            dzdx[index] += dzdy[x] / poolSize;
          }
        }
      }
    }
    data += dataSize;
    dzdx += dataSize;
    dzdy += pooledSize;
  }
}

template<typename T>
void pooling_backward_cpu_fast(T* dzdx,
                                   T const* data,
                                   T const* dzdy,
                                   int const* indices,
                                   PoolMethod method,
                                   size_t dataSize,
                                   size_t depth,
                                   size_t windowSize,
                                   size_t pooledSize)
{
#define MAX_POOL_BACK_CPU(_windowSize) case _windowSize: \
  max_pooling_backward_cpu_fast_internal<T, _windowSize>\
    (dzdx, data, dzdy, indices, dataSize, depth, pooledSize); break
#define AVG_POOL_BACK_CPU(_windowSize) case _windowSize: \
  avg_pooling_backward_cpu_fast_internal<T, _windowSize>\
    (dzdx, data, dzdy, indices, dataSize, depth, pooledSize); break

  switch (method) {
    case NN_POOL_MAX:
      switch (windowSize) {
        MAX_POOL_BACK_CPU(1);
        MAX_POOL_BACK_CPU(4);
        MAX_POOL_BACK_CPU(9);
        MAX_POOL_BACK_CPU(16);
        MAX_POOL_BACK_CPU(25);
        MAX_POOL_BACK_CPU(36);
        MAX_POOL_BACK_CPU(49);
        default:
          max_pooling_backward_cpu_fast_internal_2<T>
            (dzdx, data, dzdy, indices, dataSize, depth, windowSize, pooledSize);
          break;
      }
      break;
    case NN_POOL_AVG:
      switch (windowSize) {
        AVG_POOL_BACK_CPU(1);
        AVG_POOL_BACK_CPU(4);
        AVG_POOL_BACK_CPU(9);
        AVG_POOL_BACK_CPU(16);
        AVG_POOL_BACK_CPU(25);
        AVG_POOL_BACK_CPU(36);
        AVG_POOL_BACK_CPU(49);
        default:
          avg_pooling_backward_cpu_fast_internal_2<T>
            (dzdx, data, dzdy, indices, dataSize, depth, windowSize, pooledSize);
          break;
      }
      break;
    default:
      assert(false);
  }
#undef MAX_POOL_BACK_CPU
#undef AVG_POOL_BACK_CPU
}

template
void pooling_backward_cpu_fast<float>(float* dzdx,
                                      float const* data,
                                      float const* dzdy,
                                      int const* indices,
                                      PoolMethod method,
                                      size_t dataSize,
                                      size_t depth,
                                      size_t windowSize,
                                      size_t pooledSize);

template
void pooling_backward_cpu_fast<double>(double* dzdx,
                                       double const* data,
                                       double const* dzdy,
                                       int const* indices,
                                       PoolMethod method,
                                       size_t dataSize,
                                       size_t depth,
                                       size_t windowSize,
                                       size_t pooledSize);

void max_pooling_indices_cpu(int* indices,
                             int const* inindices,
                             size_t width,
                             size_t height,
                             size_t windowWidth,
                             size_t windowHeight,
                             size_t strideX,
                             size_t strideY,
                             size_t padLeft,
                             size_t padRight,
                             size_t padTop,
                             size_t padBottom) {
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  for (int y = 0; y < pooledHeight; ++y) {
    for (int x = 0; x < pooledWidth; ++x) {
      int x1 = x * (signed)strideX - (signed)padLeft ;
      int y1 = y * (signed)strideY - (signed)padTop ;
      int x2 = std::min(x1 + windowWidth, width) ;
      int y2 = std::min(y1 + windowHeight, height) ;
      x1 = std::max(x1, 0) ;
      y1 = std::max(y1, 0) ;

      // Set of unique pooling indices
      std::set<int> set;
      for (int v = y1 ; v < y2 ; ++v) {
        for (int u = x1 ; u < x2 ; ++u) {
          int inputIndex = v * width + u;
          if (inindices) {
            inputIndex = inindices[inputIndex];
          }
          set.insert(inputIndex);
        }
      }
      // Empty pooling region should be impossible, because size of padding is smaller than the pooling.
      if (set.empty()) {
        // mexPrintf("Empty pooling region encountered.");
        assert(false) ;
      }
      // Copy set of unique indices to a vector, copy the last values to fit the size of pooling region
      // The set is sorted, minimizing cache misses.
      std::vector<int> vec(set.begin(), set.end());
      int lastValue = vec.back();
      while (vec.size() < windowWidth * windowHeight) {
        vec.push_back(lastValue);
      }

      memcpy(indices + (y * pooledWidth + x) * (windowWidth * windowHeight),
             &vec[0],
             windowWidth * windowHeight * sizeof(int));
    }
  }
}

void avg_pooling_indices_cpu(int* indices,
                             int const* inindices,
                             size_t width,
                             size_t height,
                             size_t windowWidth,
                             size_t windowHeight,
                             size_t strideX,
                             size_t strideY,
                             size_t padLeft,
                             size_t padRight,
                             size_t padTop,
                             size_t padBottom) {
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;

  for (int y = 0; y < pooledHeight; ++y) {
    for (int x = 0; x < pooledWidth; ++x) {
      int x1 = x * (signed)strideX - (signed)padLeft ;
      int y1 = y * (signed)strideY - (signed)padTop ;
      int x2 = std::min(x1 + windowWidth, width) ;
      int y2 = std::min(y1 + windowHeight, height) ;
      x1 = std::max(x1, 0) ;
      y1 = std::max(y1, 0) ;

      // Set of unique pooling indices
      std::vector<int> vec;
      for (int v = y1 ; v < y2 ; ++v) {
        for (int u = x1 ; u < x2 ; ++u) {
          int inputIndex = v * width + u;
          if (inindices) {
            inputIndex = inindices[inputIndex];
          }
          vec.push_back(inputIndex);
        }
      }
      // Empty pooling region should be impossible, because size of padding is smaller than the pooling.
      if (vec.empty()) {
        // mexPrintf("Empty pooling region encountered.");
        assert(false) ;
      }
      // Sort the vector to improve cache locality
      std::sort(vec.begin(), vec.end());
      // Add "-1" to the back of the vector.
      while (vec.size() < windowWidth * windowHeight) {
        vec.push_back(-1);
      }

      memcpy(indices + (y * pooledWidth + x) * (windowWidth * windowHeight),
             &vec[0],
             windowWidth * windowHeight * sizeof(int));
    }
  }
}
