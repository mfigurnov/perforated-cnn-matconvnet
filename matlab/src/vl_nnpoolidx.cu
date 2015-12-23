/** @file vl_nnpoolidx.cpp
 ** @brief Pooling block
 ** @author Andrea Vedaldi
 ** @author Karel Lenc
 **/

/*
Copyright (C) 2014 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnhelper.h"
#include "bits/pooling.hpp"

#include <assert.h>

/* option codes */
enum {
  opt_method = 0,
  opt_stride,
  opt_pad,
  opt_verbose,
  opt_in_indices
} ;

/* options */
vlmxOption  options [] = {
  {"Method",           1,   opt_method            },
  {"Stride",           1,   opt_stride            },
  {"Pad",              1,   opt_pad               },
  {"InIndices",        1,   opt_in_indices        },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

VlEnumerator nnPoolMethodTypes [NN_POOL_METHODS_NUM] =
{
  {"Max",     (vl_index)NN_POOL_MAX     },
  {"Avg",     (vl_index)NN_POOL_AVG     },
} ;

enum {
  IN_DATA_SIZE = 0, IN_POOL_SIZE, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData inIndices;

  /* outputs */
  PackedData poolIndices ;
  PackedDataGeometry poolIndicesGeom ;

  PoolMethod method = NN_POOL_MAX;

  int dataWidth, dataHeight, dataDepth, dataSize ;
  int poolWidth ;
  int poolHeight ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;

  int inIndicesMode = 0 ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;
  VlEnumerator *pair ;

  packed_data_init_empty(&inIndices) ;
  packed_data_init_empty(&poolIndices) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  /* Throw an error if the input is not a GPU array. */
  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_method :
        pair = vlmxDecodeEnumeration(optarg, nnPoolMethodTypes, VL_TRUE) ;
        if (pair == NULL) {
          vlmxError(vlmxErrInvalidArgument, "METHOD is not a supported method.") ;
        }
        method = (PoolMethod)pair->value ;
        break;

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
            mexErrMsgTxt("PAD has neither one nor four elements.") ;
        }
        break;

      case opt_in_indices :
        if (mxGetNumberOfElements(optarg) != 0) {
          inIndicesMode = true;
          packed_data_init_with_array_int(&inIndices, optarg);
        }
        break;

      default: break ;
    }
  }

  if (!vlmxIsPlainMatrix(in[IN_DATA_SIZE],-1,-1)) {
    mexErrMsgTxt("DATA_SIZE is not a plain matrix.") ;
  }
  if (mxGetNumberOfElements(in[IN_DATA_SIZE]) != 4) {
    mexErrMsgTxt("DATA_SIZE does not have four elements.") ;
  }
  dataHeight = (int)mxGetPr(in[IN_DATA_SIZE])[0];
  dataWidth = (int)mxGetPr(in[IN_DATA_SIZE])[1];
  dataDepth = (int)mxGetPr(in[IN_DATA_SIZE])[2];
  dataSize = (int)mxGetPr(in[IN_DATA_SIZE])[3];

  if (!vlmxIsPlainMatrix(in[IN_POOL_SIZE],-1,-1)) {
    mexErrMsgTxt("POOL_SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_POOL_SIZE])) {
    case 1:
      poolHeight = mxGetPr(in[IN_POOL_SIZE])[0] ;
      poolWidth = poolHeight ;
      break ;
    case 2:
      poolHeight = mxGetPr(in[IN_POOL_SIZE])[0] ;
      poolWidth = mxGetPr(in[IN_POOL_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("POOL_SIZE has neither one nor two elements.") ;
  }

  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }

  int outputHeight = (dataHeight + (padTop+padBottom) - poolHeight)/strideY + 1;
  int outputWidth = (dataWidth + (padLeft+padRight) - poolWidth)/strideX + 1;
  packed_data_geom_init(&poolIndicesGeom,
                        mxINT32_CLASS,
                        poolHeight * poolWidth,
                        outputHeight,
                        outputWidth,
                        1) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnpoolidx: data: [%d %d %d %d], stride: [%d %d], pad: [%d %d %d %d], inIndicesMode: %d\n",
              dataHeight, dataWidth, dataSize, dataDepth,
              strideY, strideX,
              padTop, padBottom, padLeft, padRight,
              inIndicesMode) ;
    mexPrintf("vl_nnpoolidx: method: %s\n",
              vl_enumeration_get_by_value(nnPoolMethodTypes, method)->name);
    mexPrintf("vl_nnpoolidx: pooling: %d x %d\n", poolHeight, poolWidth);
    packed_data_geom_display(&poolIndicesGeom, "vl_nnpoolidx: poolIndices") ;
    if (inIndicesMode) {
      packed_data_geom_display(&inIndices.geom, "vl_nnpoolidx: inIndices") ;
    }
  }

  if (dataHeight < poolHeight || dataWidth < poolWidth) {
    mexErrMsgTxt("Pooling SIZE is larger than the DATA.") ;
  }

  if (poolHeight == 0 || poolWidth == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }

  if (strideX == 0 || strideY == 0) {
    mexErrMsgTxt("An element of STRIDE is zero.") ;
  }

  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }

  if (padLeft >= poolWidth ||
      padRight >= poolWidth ||
      padTop >= poolHeight  ||
      padBottom >= poolHeight) {
    mexErrMsgTxt("A padding value is larger or equal than the size of the pooling window.") ;
  }

  if (inIndicesMode) {
    if (inIndices.mode == matlabGpuArrayWrapper) {
      mexErrMsgTxt("ININDICES should be a CPU array.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  packed_data_init_with_geom_int(&poolIndices, false, poolIndicesGeom, false, false, 0) ;

  switch(method) {
    case NN_POOL_MAX:
      max_pooling_indices_cpu(poolIndices.memoryInt,
                              inIndicesMode ? inIndices.memoryInt : NULL,
                              dataHeight, dataWidth,
                              poolHeight,
                              poolWidth,
                              strideY,
                              strideX,
                              padTop,
                              padBottom,
                              padLeft,
                              padRight) ;
      break;
    case NN_POOL_AVG:
      avg_pooling_indices_cpu(poolIndices.memoryInt,
                              inIndicesMode ? inIndices.memoryInt : NULL,
                              dataHeight, dataWidth,
                              poolHeight,
                              poolWidth,
                              strideY,
                              strideX,
                              padTop,
                              padBottom,
                              padLeft,
                              padRight) ;
      break;
    default:
      assert(false);
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  out[OUT_RESULT] = packed_data_deinit_extracting_array(&poolIndices) ;
}
