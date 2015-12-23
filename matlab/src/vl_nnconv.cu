/** @file vl_nnconv.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
 ** @author Michael Figurnov
 **/

/*
 Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#include "bits/mexutils.h"
#include "bits/nnhelper.h"
#include "bits/im2col.hpp"

#include <assert.h>
#include <algorithm>

#include <blas.h>
#ifdef ENABLE_GPU
#include "bits/gpu.hpp"
#include <cublas_v2.h>
#endif

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_conv_indices,
  opt_microbatch_size,
  opt_der_filters,
  opt_der_biases,
  opt_verbose,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride             },
  {"Pad",              1,   opt_pad                },
  {"ConvIndices",      1,   opt_conv_indices       },
  {"MicrobatchSize",   1,   opt_microbatch_size    },
  {"DerFilters",       1,   opt_der_filters        },
  {"DerBiases",        1,   opt_der_biases         },
  {"Verbose",          0,   opt_verbose            },
  {"NoDerData",        0,   opt_no_der_data        },
  {"NoDerFilters",     0,   opt_no_der_filters     },
  {"NoDerBiases",      0,   opt_no_der_biases      },
  {0,                  0,   0                      }
} ;

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

#ifdef ENABLE_GPU
bool cublasInitialized = false ;
cublasHandle_t thisCublasHandle ;
#endif

bool persistentDataInitialized = false ;
PackedData temp ;
PackedData derOutputMasked;
PackedData outputMasked;
PackedData allOnes ;

void atExit()
{
  if (persistentDataInitialized) {
    packed_data_deinit (&temp)  ;
    packed_data_deinit (&derOutputMasked)  ;
    packed_data_deinit (&outputMasked)  ;
    packed_data_deinit (&allOnes)  ;
    persistentDataInitialized = false ;
  }
#ifdef ENABLE_GPU
  if (cublasInitialized) {
    cublasDestroy(thisCublasHandle) ;
    cublasInitialized = false ;
  }
#endif
}

/* ---------------------------------------------------------------- */
/*                                                  Dispatcher func */
/* ---------------------------------------------------------------- */

static void
sgemv_dispatch(bool gpuMode,
               char op,
               ptrdiff_t m, ptrdiff_t n,
               float alpha,
               float const * a, ptrdiff_t lda,
               float const * x, ptrdiff_t incx,
               float beta,
               float * y, ptrdiff_t incy)
{
  if (!gpuMode) {
    sgemv(&op,
          &m, &n, &alpha,
          (float*)a, &lda,
          (float*)x, &incx,
          &beta,
          y, &incy) ;
  } else {
#ifdef ENABLE_GPU
    cublasSgemv(thisCublasHandle,
                (op == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (int)m, (int)n,
                &alpha,
                a, lda,
                x, (int)incx,
                &beta,
                y, (int)incy) ;
#endif
  }
}

static void
sgemm_dispatch(bool gpuMode,
               char op1, char op2,
               ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
               float alpha,
               float const * a, ptrdiff_t lda,
               float const * b, ptrdiff_t ldb,
               float beta,
               float * c, ptrdiff_t ldc)
{
  if (!gpuMode) {
    sgemm(&op1, &op2,
          &m, &n, &k,
          &alpha,
          (float*)a, &lda,
          (float*)b, &ldb,
          &beta,
          c, &ldc) ;
  } else {
#ifdef ENABLE_GPU
    cublasSgemm(thisCublasHandle,
                (op1 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (op2 == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
                (int)m, (int)n, (int)k,
                &alpha,
                a, (int)lda,
                b, (int)ldb,
                &beta,
                c, (int)ldc);
#endif
  }
}

static void
im2col_dispatch(bool gpuMode,
                float* stacked,
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
                size_t padBottom)
{
  if (!gpuMode) {
    im2col_cpu<float>(stacked,
                      data,
                      width,
                      height,
                      depth,
                      windowWidth,
                      windowHeight,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
  } else {
#ifdef ENABLE_GPU
    im2col_gpu<float>(stacked,
                      data,
                      width,
                      height,
                      depth,
                      windowWidth,
                      windowHeight,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
#endif
  }
}

static void
im2col_indexed_dispatch(bool gpuMode,
                        float* stacked,
                        float const* data,
                        int const* im2colIndices,
                        int im2colIndicesLength,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t size,
                        size_t windowWidth,
                        size_t windowHeight)
{
  if (!gpuMode) {
    im2col_indexed_cpu<float>(stacked,
                              data,
                              im2colIndices,
                              im2colIndicesLength,
                              width,
                              height,
                              depth,
                              size,
                              windowWidth,
                              windowHeight);
  } else {
#ifdef ENABLE_GPU
    im2col_indexed_gpu<float>(stacked,
                              data,
                              im2colIndices,
                              im2colIndicesLength,
                              width,
                              height,
                              depth,
                              size,
                              windowWidth,
                              windowHeight);
#endif
  }
}

static void
col2im_dispatch(bool gpuMode,
                float* data,
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
                size_t padBottom)
{
  if (!gpuMode) {
    col2im_cpu<float>(data,
                      stacked,
                      width,
                      height,
                      depth,
                      windowWidth,
                      windowHeight,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
  } else {
#ifdef ENABLE_GPU
    col2im_gpu<float>(data,
                      stacked,
                      width,
                      height,
                      depth,
                      windowWidth,
                      windowHeight,
                      strideX,
                      strideY,
                      padLeft,
                      padRight,
                      padTop,
                      padBottom) ;
#endif
  }
}

static void
col2im_indexed_dispatch(bool gpuMode,
                        float* data,
                        float const* stacked,
                        int const* im2colIndices,
                        int im2colIndicesLength,
                        size_t width,
                        size_t height,
                        size_t depth,
                        size_t size,
                        size_t windowWidth,
                        size_t windowHeight)
{
  if (!gpuMode) {
    col2im_indexed_cpu(data,
                       stacked,
                       im2colIndices,
                       im2colIndicesLength,
                       width,
                       height,
                       depth,
                       size,
                       windowWidth,
                       windowHeight);
  } else {
#ifdef ENABLE_GPU
    col2im_indexed_gpu(data,
                       stacked,
                       im2colIndices,
                       im2colIndicesLength,
                       width,
                       height,
                       depth,
                       size,
                       windowWidth,
                       windowHeight) ;
#endif
  }
}

static void
transpose23_dispatch(bool gpuMode,
                     float* transposed,
                     float const* data,
                     size_t d1,
                     size_t d2,
                     size_t d3)
{
  if (!gpuMode) {
    transpose23_cpu(transposed, data, d1, d2, d3);
  } else {
#ifdef ENABLE_GPU
    transpose23_gpu(transposed, data, d1, d2, d3) ;
#endif
  }
}

static void
copy_dispatch(bool gpuMode,
              float * dest,
              float const * src,
              size_t numElements)
{
  if (!gpuMode) {
    memcpy(dest, src, numElements * sizeof(float)) ;
  } else {
#ifdef ENABLE_GPU
    cudaMemcpy(dest, src, numElements * sizeof(float), cudaMemcpyDeviceToDevice) ;
#endif
  }
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData data ;
  PackedData filters ;
  PackedData biases ;
  PackedData derOutput ;
  PackedData convIndices ;
  PackedData derFiltersInit ;
  PackedData derBiasesInit ;

  /* outputs */
  PackedData output ;
  PackedData derData  ;
  PackedData derFilters ;
  PackedData derBiases ;

  PackedDataGeometry outputGeom ;
  PackedDataGeometry derDataGeom  ;
  PackedDataGeometry derFiltersGeom ;
  PackedDataGeometry derBiasesGeom ;
  PackedDataGeometry tempGeom ;
  PackedDataGeometry derOutputMaskedGeom ;
  PackedDataGeometry outputMaskedGeom ;
  PackedDataGeometry allOnesGeom ;

  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  int numGroups = 1 ;
  int microbatchSize = 1 ;

#if ENABLE_GPU
  cublasStatus_t stat;
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;
  bool hasFilters = false ;
  bool hasBiases = false ;
  bool fullyConnectedMode = false ;
  bool is_1x1 = false ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computeDerBiases = true ;
  bool convIndicesMode = false;
  bool derFiltersInitialized = false ;
  bool derBiasesInitialized = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  packed_data_init_empty(&data) ;
  packed_data_init_empty(&filters) ;
  packed_data_init_empty(&biases) ;
  packed_data_init_empty(&derOutput) ;
  packed_data_init_empty(&convIndices) ;
  packed_data_init_empty(&output) ;
  packed_data_init_empty(&derData) ;
  packed_data_init_empty(&derFilters) ;
  packed_data_init_empty(&derBiases) ;
  packed_data_init_empty(&derFiltersInit) ;
  packed_data_init_empty(&derBiasesInit) ;
  if (!persistentDataInitialized) {
    packed_data_init_empty(&temp) ;
    packed_data_init_empty(&outputMasked) ;
    packed_data_init_empty(&allOnes) ;
    persistentDataInitialized = true ;
  }

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    mexErrMsgTxt("There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_conv_indices :
        if (mxGetNumberOfElements(optarg) != 0) {
          convIndicesMode = true;
          packed_data_init_with_array_int(&convIndices, optarg);
        }
        break;

      case opt_microbatch_size :
        if (mxGetNumberOfElements(optarg) == 1) {
          microbatchSize = (int)mxGetPr(optarg)[0] ;
        }
        break;

      case opt_der_filters :
        if (mxGetNumberOfElements(optarg) != 0) {
          derFiltersInitialized = true;
          packed_data_init_with_array(&derFiltersInit, optarg);
        }
        break;

      case opt_der_biases :
        if (mxGetNumberOfElements(optarg) != 0) {
          derBiasesInitialized = true;
          packed_data_init_with_array(&derBiasesInit, optarg);
        }
        break;

      case opt_no_der_data :
        computeDerData = VL_FALSE ;
        break ;

      case opt_no_der_filters :
        computeDerFilters = VL_FALSE ;
        break ;

      case opt_no_der_biases :
        computeDerBiases = VL_FALSE ;
        break ;

      default: break ;
    }
  }

  packed_data_init_with_array(&data, in[IN_DATA]) ;
  packed_data_init_with_array(&filters, in[IN_FILTERS]) ;
  packed_data_init_with_array(&biases, in[IN_BIASES]) ;
  if (backMode) { packed_data_init_with_array(&derOutput, in[IN_DEROUTPUT]) ; }

#if ENABLE_GPU
  gpuMode = (data.mode == matlabGpuArrayWrapper) ;
  if (gpuMode) {
    mxInitGPU() ;
    if (!cublasInitialized) {
      stat = cublasCreate(&thisCublasHandle) ;
      if (stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("Could not initialize cuBLAS.") ;
      }
      cublasInitialized = true ;
    }
  }
#endif

  hasFilters = filters.geom.numElements > 0 ;
  hasBiases = biases.geom.numElements > 0 ;

  /* check for GPU/data class consistency */
  if (! hasFilters) {
    mexErrMsgTxt("FILTERS is empty.") ;
  }
  if (! packed_data_are_compatible(&data, &filters)) {
    mexErrMsgTxt("DATA and FILTERS are not both CPU or GPU arrays.") ;
  }
  if (hasBiases && ! packed_data_are_compatible(&data, &biases)) {
    mexErrMsgTxt("DATA and BIASES are not both CPU or GPU arrays.") ;
  }
  if (backMode && ! packed_data_are_compatible(&data, &derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }
  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (filters.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTERS is not of class SINGLE.");
  }
  if (hasBiases && (biases.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("BIASES is not of class SINGLE.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
  }

  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (convIndicesMode && ! packed_data_are_compatible(&data, &convIndices)) {
    mexErrMsgTxt("DATA and CONVINDICES are not both CPU or GPU arrays.") ;
  }
  if (convIndicesMode && (convIndices.geom.classID != mxINT32_CLASS)) {
    mexErrMsgTxt("CONVINDICES is not of class INT32.");
  }

  if (convIndicesMode) {
    packed_data_geom_init(&outputGeom,
                          mxSINGLE_CLASS,
                          convIndices.geom.height,
                          convIndices.geom.width,
                          filters.geom.size,
                          data.geom.size) ;
  } else {
    packed_data_geom_init(&outputGeom,
                          mxSINGLE_CLASS,
                          (data.geom.height + (padTop+padBottom) - filters.geom.height)/strideY + 1,
                          (data.geom.width + (padLeft+padRight) - filters.geom.width)/strideX + 1,
                          filters.geom.size,
                          data.geom.size) ;
  }

  /* grouped filters */
  numGroups = data.geom.depth / filters.geom.depth ;

  /* if the output is 1x1 pixels, then there is no need to actually
   call im2col as it does not do anything
   */
  fullyConnectedMode = (!convIndicesMode &&
                        outputGeom.height == 1 &&
                        outputGeom.width == 1 &&
                        padTop == 0 &&
                        padBottom == 0 &&
                        padLeft == 0 &&
                        padRight == 0 &&
                        numGroups == 1) ;
  is_1x1 = (!convIndicesMode &&
            filters.geom.height == 1 &&
            filters.geom.width == 1 &&
            strideY == 1 &&
            strideX == 1 &&
            padTop == 0 &&
            padBottom == 0 &&
            padLeft == 0 &&
            padRight == 0);

  if (convIndicesMode) {
    if (convIndices.geom.depth != filters.geom.height*filters.geom.width) {
      mexErrMsgTxt("CONVINDICES depth is not compatible with filters.");
    }

    if (convIndices.geom.size != 1 && convIndices.geom.size != data.geom.size) {
      mexErrMsgTxt("CONVINDICES size should be equal either one, or the number of input images.");
    }
  }

  if (!is_1x1) {
    packed_data_geom_init
    (&tempGeom, mxSINGLE_CLASS,
     outputGeom.height,
     outputGeom.width,
     filters.geom.height*filters.geom.width*filters.geom.depth*numGroups,
     microbatchSize) ;
  } else {
    packed_data_geom_init (&tempGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  if (convIndicesMode) {
    packed_data_geom_init
    (&outputMaskedGeom, mxSINGLE_CLASS,
     outputGeom.height,
     outputGeom.width,
     filters.geom.size,
     microbatchSize) ;
  } else {
    packed_data_geom_init (&outputMaskedGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  if (false) {
    packed_data_geom_init (&derOutputMaskedGeom, mxSINGLE_CLASS,
                           outputGeom.height,
                           outputGeom.width,
                           filters.geom.size,
                           microbatchSize) ;
  } else {
    packed_data_geom_init (&derOutputMaskedGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  derDataGeom = data.geom ;
  derFiltersGeom = filters.geom ;
  if (hasBiases) {
    if (fullyConnectedMode) {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             1, 1,
                             1, data.geom.size) ;
    } else {
      packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                             outputGeom.height,
                             outputGeom.width,
                             1, microbatchSize) ;
    }
    derBiasesGeom = biases.geom ;
  } else {
    packed_data_geom_init (&allOnesGeom, mxSINGLE_CLASS,
                           0, 0, 0, 0) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnconv: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnconv: stride: [%d %d], pad: [%d %d %d %d], numGroups: %d, has bias: %d, fully connected: %d, 1x1: %d, conv indices: %d, microbatchSize: %d\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight,
              numGroups, hasBiases, fullyConnectedMode, is_1x1, convIndicesMode,
              microbatchSize) ;
    packed_data_geom_display(&data.geom, "vl_nnconv: data") ;
    packed_data_geom_display(&filters.geom, "vl_nnconv: filters") ;
    if (hasBiases) { packed_data_geom_display(&biases.geom, "vl_nnconv: biases") ; }
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnconv: derOutput") ;
      packed_data_geom_display(&derOutputMaskedGeom, "vl_nnconv: derOutputMasked") ;
      packed_data_geom_display(&derOutputMasked.geom, "vl_nnconv: derOutputMasked (cached)") ;
      packed_data_geom_display(&derDataGeom, "vl_nnconv: derData") ;
      packed_data_geom_display(&derFiltersGeom, "vl_nnconv: derFilters") ;
      if (hasBiases) { packed_data_geom_display(&derBiasesGeom, "vl_nnconv: derBiases") ; }
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnconv: output") ;
    }
    packed_data_geom_display(&tempGeom, "vl_nnconv: temp") ;
    packed_data_geom_display(&temp.geom, "vl_nnconv: temp (cached)") ;
    packed_data_geom_display(&outputMaskedGeom, "vl_nnconv: outputMasked") ;
    packed_data_geom_display(&outputMasked.geom, "vl_nnconv: outputMasked (cached)") ;
    packed_data_geom_display(&allOnesGeom, "vl_nnconv: allOnes") ;
    packed_data_geom_display(&allOnes.geom, "vl_nnconv: allOnes (cached)") ;
    if (convIndicesMode) {
      packed_data_geom_display(&convIndices.geom, "vl_nnconv: convIndices") ;
    }
  }

  if (backMode) {
    if (derOutput.geom.height != outputGeom.height ||
        derOutput.geom.width != outputGeom.width ||
        derOutput.geom.depth != filters.geom.size ||
        derOutput.geom.size != data.geom.size)
    {
      mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and FILTERS.") ;
    }
  }

  if (numGroups * filters.geom.depth != data.geom.depth) {
    mexErrMsgTxt("The filter depth does not divide the image depth.") ;
  }

  if (filters.geom.size % numGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the total number of filters.") ;
  }

  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }

  if (outputGeom.height == 0 || outputGeom.width == 0) {
    mexErrMsgTxt("FILTERS are larger than the DATA (including padding).") ;
  }

  if (filters.geom.height == 0 || filters.geom.width == 0 || filters.geom.depth == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  if (hasBiases) {
    if (biases.geom.numElements != filters.geom.size) {
      mexErrMsgTxt("The number of elements of BIASES is not the same as the number of filters.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  /* auxiliary buffers */
  if (hasBiases) {
    if (allOnes.memorySize < allOnesGeom.numElements * sizeof(float) ||
        (allOnes.mode == matlabGpuArray || allOnes.mode == matlabGpuArrayWrapper) != gpuMode) {
      packed_data_deinit (&allOnes) ;
      packed_data_init_with_geom (&allOnes, gpuMode, allOnesGeom, true, true, 1.0f) ;
    }
  }
  if (!fullyConnectedMode) {
    if (temp.memorySize < tempGeom.numElements * sizeof(float) ||
        (temp.mode == matlabGpuArray || temp.mode == matlabGpuArrayWrapper) != gpuMode) {
      packed_data_deinit (&temp) ;
      packed_data_init_with_geom (&temp, gpuMode, tempGeom, true, false, 0);
    }
  }
  if (derOutputMasked.memorySize < derOutputMaskedGeom.numElements * sizeof(float) ||
      (derOutputMasked.mode == matlabGpuArray || derOutputMasked.mode == matlabGpuArrayWrapper) != gpuMode) {
    packed_data_deinit (&derOutputMasked) ;
    packed_data_init_with_geom (&derOutputMasked, gpuMode, derOutputMaskedGeom, true, false, 0);
  }
  if (outputMasked.memorySize < outputMaskedGeom.numElements * sizeof(float) ||
      (outputMasked.mode == matlabGpuArray || outputMasked.mode == matlabGpuArrayWrapper) != gpuMode) {
    packed_data_deinit (&outputMasked) ;
    packed_data_init_with_geom (&outputMasked, gpuMode, outputMaskedGeom, true, false, 0);
  }
  if (!backMode) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, false, 0) ;
  } else {
    if (computeDerData) {
      packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, false, 0) ;
    }
    if (computeDerFilters) {
      packed_data_init_with_geom(&derFilters, gpuMode, derFiltersGeom, false, false, 0) ;
      if (derFiltersInitialized) {
        copy_dispatch(gpuMode, derFilters.memory, derFiltersInit.memory, derFilters.geom.numElements);;
      }
    }
    if (computeDerBiases && hasBiases) {
      packed_data_init_with_geom(&derBiases, gpuMode, derBiasesGeom, false, false, 0) ;
      if (derFiltersInitialized) {
        copy_dispatch(gpuMode, derBiases.memory, derBiasesInit.memory, derBiases.geom.numElements);;
      }
    }
  }

  if (fullyConnectedMode) {
    float alpha = 1 ;
    float beta = 0 ;
    ptrdiff_t filtersVolume = filters.geom.height*filters.geom.width*filters.geom.depth ;
    /* note: fullyConnectedMode also guarantees no padding, num filter groups = 1 */

    /* optimise fully-connected mode case */
    if (!backMode) {
      if (data.geom.size == 1) {
        /* one image in the stack */
        sgemv_dispatch(gpuMode, 't',
                       filtersVolume, filters.geom.size,
                       alpha,
                       filters.memory, filtersVolume,
                       data.memory, 1,
                       beta,
                       output.memory, 1) ;
      } else {
        /* multiple images in the stack */
        sgemm_dispatch(gpuMode, 't', 'n',
                       filters.geom.size, data.geom.size, filtersVolume,
                       alpha,
                       filters.memory, filtersVolume,
                       data.memory, filtersVolume,
                       beta,
                       output.memory, filters.geom.size) ;
      }
      if (hasBiases) {
        float beta = 1 ;
        ptrdiff_t q = 1 ;
        sgemm_dispatch(gpuMode, 'n', 'n',
                       filters.geom.size, data.geom.size, q,
                       alpha,
                       biases.memory, filters.geom.size,
                       allOnes.memory, q,
                       beta,
                       output.memory, filters.geom.size) ;
      }
    } else {
      /* back mode */
      if (computeDerFilters) {
        sgemm_dispatch(gpuMode, 'n', 't',
                       filtersVolume, filters.geom.size, data.geom.size,
                       alpha,
                       data.memory, filtersVolume,
                       derOutput.memory, filters.geom.size,
                       (float)(derFiltersInitialized > 0),
                       derFilters.memory, filtersVolume) ;
      }
      if (computeDerBiases && hasBiases) {
        ptrdiff_t q = 1 ;
        sgemm_dispatch(gpuMode, 'n', 't',
                       q, filters.geom.size, data.geom.size,
                       alpha,
                       allOnes.memory, q,
                       derOutput.memory, filters.geom.size,
                       (float)(derBiasesInitialized > 0),
                       derBiases.memory, q) ;
      }
      if (computeDerData) {
        sgemm_dispatch(gpuMode, 'n', 'n',
                       filtersVolume, data.geom.size, filters.geom.size,
                       alpha,
                       filters.memory, filtersVolume,
                       derOutput.memory, filters.geom.size,
                       beta,
                       derData.memory, filtersVolume) ;
      }
    }
  } else if (convIndicesMode) {
    // microbatchSize specifies the number of images to stack for GEMM
    const int numMicrobatches = (data.geom.size + microbatchSize - 1) / microbatchSize;
    for (int microbatchIdx = 0; microbatchIdx < numMicrobatches; ++microbatchIdx) {
      int image = microbatchIdx * microbatchSize;
      int numImages = (microbatchIdx != numMicrobatches - 1) ? microbatchSize : (data.geom.size - image);

      ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
      ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
      ptrdiff_t derDataOffset = (derData.geom.height*derData.geom.width*derData.geom.depth) * image ;
      ptrdiff_t derOutputOffset = (derOutput.geom.height*derOutput.geom.width*derOutput.geom.depth) * image ;
      ptrdiff_t m = outputGeom.height * outputGeom.width ; /* num output pixels */
      ptrdiff_t numRows = m * numImages ;
      ptrdiff_t n = filters.geom.size/numGroups ; /* num filters per group */
      ptrdiff_t k = filters.geom.height*filters.geom.width*filters.geom.depth ; /* filter volume */

      if (backMode) {
        if (numImages > 1) {
          transpose23_dispatch(gpuMode,
                               outputMasked.memory,
                               derOutput.memory + derOutputOffset,
                               m, derOutput.geom.depth, numImages) ;
        }

        float *curDerOutputMemory = numImages > 1 ? outputMasked.memory : derOutput.memory + derOutputOffset;

        /* compute derFilters dz/dF */
        if (computeDerFilters) {
          im2col_indexed_dispatch(gpuMode,
                                  temp.memory,
                                  data.memory + dataOffset,
                                  convIndices.memoryInt,
                                  convIndices.geom.numElements,
                                  data.geom.height, data.geom.width, data.geom.depth, numImages,
                                  filters.geom.height, filters.geom.width) ;
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = numRows * k * g ;
            ptrdiff_t derOutputGrpOffset = numRows * n * g ;
            float alpha = 1 ;
            float beta = (image > 0 || derFiltersInitialized) ; /* this saves init. the output array with 0 */
            sgemm_dispatch(gpuMode, 't', 'n',
                           k, n, numRows,
                           alpha,
                           temp.memory + tempGrpOffset, numRows,
                           curDerOutputMemory + derOutputGrpOffset, numRows,
                           beta,
                           derFilters.memory + filterGrpOffset, k) ;
          }
        }

        /* compute derData dz/dbias */
        if (computeDerBiases & hasBiases) {
          sgemv_dispatch(gpuMode, 't',
                         numRows, filters.geom.size,
                         1, /* alpha */
                         curDerOutputMemory, numRows,
                         allOnes.memory, 1,
                         (float)(image > 0 || derBiasesInitialized), /* beta */
                         derBiases.memory, 1) ;
        }

        /* compute derData dz/dx */
        if (computeDerData) {
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = numRows * k * g ;
            ptrdiff_t derOutputGrpOffset = numRows * n * g ;
            float alpha = 1 ;
            float beta = 0 ;
            sgemm_dispatch(gpuMode, 'n', 't',
                           numRows, k, n,
                           alpha,
                           curDerOutputMemory + derOutputGrpOffset, numRows,
                           filters.memory + filterGrpOffset, k,
                           beta,
                           temp.memory + tempGrpOffset,
                           numRows) ;
          }
          col2im_indexed_dispatch(gpuMode,
                                  derData.memory + derDataOffset,
                                  temp.memory,
                                  convIndices.memoryInt,
                                  convIndices.geom.numElements,
                                  data.geom.height, data.geom.width, data.geom.depth, numImages,
                                  filters.geom.height, filters.geom.width);
        }
      } else {
        float *curOutputMemory = numImages > 1 ? outputMasked.memory : output.memory + outputOffset;

        im2col_indexed_dispatch(gpuMode,
                                temp.memory,
                                data.memory + dataOffset,
                                convIndices.memoryInt,
                                convIndices.geom.numElements,
                                data.geom.height, data.geom.width, data.geom.depth, numImages,
                                filters.geom.height, filters.geom.width) ;
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = numRows * k * g ;
          ptrdiff_t outputGrpOffset = numRows * n * g  ;
          float alpha = 1 ;
          float beta = 0 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         numRows, n, k,
                         alpha,
                         temp.memory + tempGrpOffset, numRows,
                         filters.memory + filterGrpOffset, k,
                         beta,
                         curOutputMemory + outputGrpOffset,
                         numRows) ;
        }
        if (hasBiases) {
          float alpha = 1 ;
          float beta = 1 ;
          ptrdiff_t q = 1 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         numRows, biases.geom.numElements, q,
                         alpha,
                         allOnes.memory, numRows,
                         biases.memory, q,
                         beta,
                         curOutputMemory,
                         numRows) ;
        }

        if (numImages > 1) {
          transpose23_dispatch(gpuMode,
                               output.memory + outputOffset,
                               outputMasked.memory,
                               m, numImages, output.geom.depth) ;
        }
      }
    }
  } else {
    // This branch catches corner cases: 1x1 convolutions (skipping im2col/col2im), and when
    // vl_nnconv called without convIndices.
    // It can be merged with the previous branch, but the number of conditionals inside is already
    // way too high.
    for (int image = 0 ; image < data.geom.size ; ++image) {
      /*
       temp (phi(x)): m x k
       filters, derFilters: k x n (for one group of filters)
       derOutput (dzdy) : m x n (for one group of filters)
       res (y) : m x n (for one group of filters)
       */
      ptrdiff_t dataOffset = (data.geom.height*data.geom.width*data.geom.depth) * image ;
      ptrdiff_t outputOffset = (output.geom.height*output.geom.width*output.geom.depth) * image ;
      ptrdiff_t derDataOffset = (derData.geom.height*derData.geom.width*derData.geom.depth) * image ;
      ptrdiff_t derOutputOffset = (derOutput.geom.height*derOutput.geom.width*derOutput.geom.depth) * image ;
      ptrdiff_t m = outputGeom.height * outputGeom.width ; /* num output pixels */
      ptrdiff_t n = filters.geom.size/numGroups ; /* num filters per group */
      ptrdiff_t k = filters.geom.height*filters.geom.width*filters.geom.depth ; /* filter volume */

      float* tempMemory;

      if (backMode) {
        /* ---------------------------------------------------------- */
        /*                                              Backward mode */
        /* ---------------------------------------------------------- */

        /* compute derFilters dz/dF */
        if (computeDerFilters) {
          if (!is_1x1) {
            im2col_dispatch(gpuMode,
                            temp.memory,
                            data.memory + dataOffset,
                            data.geom.height, data.geom.width, data.geom.depth,
                            filters.geom.height, filters.geom.width,
                            strideY, strideX,
                            padTop, padBottom, padLeft, padRight) ;
            tempMemory = temp.memory;
          } else {
            tempMemory = data.memory + dataOffset;
          }
          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = (image > 0 || derFiltersInitialized) ; /* this saves init. the output array with 0 */
            sgemm_dispatch(gpuMode, 't', 'n',
                           k, n, m,
                           alpha,
                           tempMemory + tempGrpOffset, m,
                           derOutput.memory + derOutputOffset + derOutputGrpOffset, m,
                           beta,
                           derFilters.memory + filterGrpOffset, k) ;
          }
        }

        /* compute derData dz/dbias */
        if (computeDerBiases & hasBiases) {
          sgemv_dispatch(gpuMode, 't',
                         m, filters.geom.size,
                         1, /* alpha */
                         derOutput.memory + derOutputOffset, m,
                         allOnes.memory, 1,
                         (float)(image > 0 || derBiasesInitialized), /* beta */
                         derBiases.memory, 1) ;
        }

        /* compute derData dz/dx */
        if (computeDerData) {
          if (!is_1x1) {
            tempMemory = temp.memory;
          } else {
            tempMemory = derData.memory + derDataOffset;
          }

          for (int g = 0 ; g < numGroups ; ++ g) {
            ptrdiff_t filterGrpOffset = k * n * g ;
            ptrdiff_t tempGrpOffset = m * k * g ;
            ptrdiff_t derOutputGrpOffset = m * n * g ;
            float alpha = 1 ;
            float beta = 0 ;
            sgemm_dispatch(gpuMode, 'n', 't',
                           m, k, n,
                           alpha,
                           derOutput.memory + derOutputOffset + derOutputGrpOffset, m,
                           filters.memory + filterGrpOffset, k,
                           beta,
                           tempMemory + tempGrpOffset,
                           m) ;
          }
          if (!is_1x1) {
            col2im_dispatch(gpuMode,
                            derData.memory + derDataOffset,
                            temp.memory,
                            data.geom.height, data.geom.width, data.geom.depth,
                            filters.geom.height, filters.geom.width,
                            strideY, strideX,
                            padTop, padBottom, padLeft, padRight) ;
          }
        }
      } else {
        /* ---------------------------------------------------------- */
        /*                                               Forward mode */
        /* ---------------------------------------------------------- */
        if (!is_1x1) {
          im2col_dispatch(gpuMode,
                          temp.memory,
                          data.memory + dataOffset,
                          data.geom.height, data.geom.width, data.geom.depth,
                          filters.geom.height, filters.geom.width,
                          strideY, strideX,
                          padTop, padBottom, padLeft, padRight) ;
          tempMemory = temp.memory;
        } else {
          tempMemory = data.memory + dataOffset;
        }
        for (int g = 0 ; g < numGroups ; ++ g) {
          ptrdiff_t filterGrpOffset = k * n * g ;
          ptrdiff_t tempGrpOffset = m * k * g ;
          ptrdiff_t outputGrpOffset = m * n * g  ;
          float alpha = 1 ;
          float beta = 0 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         m, n, k,
                         alpha,
                         tempMemory + tempGrpOffset, m,
                         filters.memory + filterGrpOffset, k,
                         beta,
                         output.memory + outputOffset + outputGrpOffset, m) ;
        }
        if (hasBiases) {
          float alpha = 1 ;
          float beta = 1 ;
          ptrdiff_t q = 1 ;
          sgemm_dispatch(gpuMode, 'n', 'n',
                         m, biases.geom.numElements, q,
                         alpha,
                         allOnes.memory, m,
                         biases.memory, q,
                         beta,
                         output.memory + outputOffset, m) ;
        }
      }
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  packed_data_deinit(&data) ;
  packed_data_deinit(&filters) ;
  packed_data_deinit(&biases) ;
  if (convIndicesMode) {
    packed_data_deinit(&convIndices);
  }
  if (backMode) {
    packed_data_deinit(&derOutput) ;
    out[OUT_RESULT] = (computeDerData) ? packed_data_deinit_extracting_array(&derData) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERFILTERS] =(computeDerFilters)? packed_data_deinit_extracting_array(&derFilters) : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERBIASES] = (computeDerBiases & hasBiases) ? packed_data_deinit_extracting_array(&derBiases) : mxCreateDoubleMatrix(0,0,mxREAL) ;
  } else {
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&output) ;
  }
  packed_data_deinit(&derFiltersInit) ;
  packed_data_deinit(&derBiasesInit) ;
}
