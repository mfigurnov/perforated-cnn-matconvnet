/** @file im2col.hpp
 ** @brief Image to columns and back
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __matconv__im2col__
#define __matconv__im2col__

#include <assert.h>
#include <stddef.h>
#include "mex.h"

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
                size_t padBottom) ;

template <typename T>
void im2col_indexed_cpu(T* stacked,
                        T const* data,
                        int const* indices,
                        int indicesSize,
                        int width,
                        int height,
                        int depth,
                        int size,
                        int windowWidth,
                        int windowHeight);

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
                size_t padBottom) ;

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
                        int windowHeight) ;

template<typename T>
void transpose23_cpu(T* transposed,
                     T const* data,
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
                      int padBottom);

#ifdef ENABLE_GPU
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
                size_t padBottom) ;

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
                        size_t windowHeight) ;

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
                size_t padBottom) ;

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
                        size_t windowHeight) ;

template <typename T>
void transpose23_gpu(T* transposed,
                     T const* data,
                     size_t d1,
                     size_t d2,
                     size_t d3) ;

#endif

#endif /* defined(__matconv__im2col__) */
