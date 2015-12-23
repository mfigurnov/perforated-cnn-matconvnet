/** @file vl_nnconvperf.cu
 ** @brief Convolution block
 ** @author Andrea Vedaldi
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

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_in_indices,
  opt_mask_indices,
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride             },
  {"Pad",              1,   opt_pad                },
  {"InIndices",        1,   opt_in_indices         },
  {"MaskIndices",      1,   opt_mask_indices       },
  {"Verbose",          0,   opt_verbose            },
  {0,                  0,   0                      }
} ;

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA_SIZE = 0, IN_FILTERS_SIZE, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData inIndices ;
  PackedData maskIndices ;

  /* outputs */
  PackedData convIndices ;

  PackedDataGeometry convIndicesGeom ;

  int dataWidth, dataHeight, dataDepth, dataSize ;
  int filtersWidth, filtersHeight, filtersDepth, filtersSize ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  int numGroups = 1 ;

  int maskIndicesLength ;

  bool inMaskMode = false;
  bool maskMode = false;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  packed_data_init_empty(&inIndices) ;
  packed_data_init_empty(&maskIndices) ;
  packed_data_init_empty(&convIndices) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 2) {
    mexErrMsgTxt("There are less than two arguments.") ;
  }

  next = 2;

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

      case opt_in_indices :
        if (mxGetNumberOfElements(optarg) != 0) {
          inMaskMode = true;
          packed_data_init_with_array_int(&inIndices, optarg);
        }
        break;

      case opt_mask_indices :
        if (mxGetNumberOfElements(optarg) != 0) {
          maskMode = true;
          packed_data_init_with_array_int(&maskIndices, optarg);
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

  if (!vlmxIsPlainMatrix(in[IN_FILTERS_SIZE],-1,-1)) {
    mexErrMsgTxt("FILTERS_SIZE is not a plain matrix.") ;
  }
  if (mxGetNumberOfElements(in[IN_FILTERS_SIZE]) != 4) {
    mexErrMsgTxt("FILTERS_SIZE does not have four elements.") ;
  }
  filtersHeight = (int)mxGetPr(in[IN_FILTERS_SIZE])[0];
  filtersWidth = (int)mxGetPr(in[IN_FILTERS_SIZE])[1];
  filtersDepth = (int)mxGetPr(in[IN_FILTERS_SIZE])[2];
  filtersSize = (int)mxGetPr(in[IN_FILTERS_SIZE])[3];

  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }

  if (inMaskMode && (inIndices.geom.classID != mxINT32_CLASS)) {
    mexErrMsgTxt("ININDICES is not of class INT32.");
  }

  if (maskMode && (maskIndices.geom.classID != mxINT32_CLASS)) {
    mexErrMsgTxt("MASKINDICES is not of class INT32.");
  }

  int outputGeomHeight = (dataHeight + (padTop+padBottom) - filtersHeight)/strideY + 1;
  int outputGeomWidth = (dataWidth + (padLeft+padRight) - filtersWidth)/strideX + 1;

  /* grouped filters */
  numGroups = dataDepth / filtersDepth ;

  if (inMaskMode) {
    if (inIndices.geom.height != dataHeight ||
        inIndices.geom.width != dataWidth) {
      mexErrMsgTxt("ININDICES height and width are not compatible with data.") ;
    }

    if (inIndices.geom.depth != 1) {
      mexErrMsgTxt("ININDICES depth should be equal to one.") ;
    }

    if (inIndices.geom.size != 1 && inIndices.geom.size != dataSize) {
      mexErrMsgTxt("ININDICES size should be equal either one, or the number of input images.");
    }
  }

  if (maskMode) {
    if (maskIndices.geom.width != 1 ||
        maskIndices.geom.depth != 1) {
      mexErrMsgTxt("MASKINDICES width and depth should be equal to one.") ;
    }

    if (maskIndices.geom.size != 1 && maskIndices.geom.size != dataSize) {
      mexErrMsgTxt("MASKINDICES size should be equal either one, or the number of input images.");
    }

    maskIndicesLength = maskIndices.geom.height;
  }

  packed_data_geom_init(&convIndicesGeom,
                        mxINT32_CLASS,
                        maskMode ? maskIndicesLength : outputGeomHeight,
                        maskMode ? 1 : outputGeomWidth,
                        filtersHeight * filtersWidth,
                        1) ;

  if (verbosity > 0) {
    mexPrintf("vl_nnconvidx: stride: [%d %d], pad: [%d %d %d %d], numGroups: %d, has input mask: %d, has mask: %d\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight,
              numGroups, inMaskMode, maskMode) ;
    mexPrintf("vl_nnconvidx: data: [%d %d %d %d], filters: [%d %d %d %d]\n",
              dataHeight, dataWidth, dataSize, dataDepth,
              filtersHeight, filtersWidth, filtersSize, filtersDepth);
    packed_data_geom_display(&convIndicesGeom, "vl_nnconvidx: convIndices") ;
    if (inMaskMode) {
      packed_data_geom_display(&inIndices.geom, "vl_nnconvidx: inIndices") ;
    }
    if (maskMode) {
      packed_data_geom_display(&maskIndices.geom, "vl_nnconvidx: maskIndices") ;
    }
  }

  if (numGroups * filtersDepth != dataDepth) {
    mexErrMsgTxt("The filter depth does not divide the image depth.") ;
  }

  if (filtersSize % numGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the total number of filters.") ;
  }

  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }

  if (dataHeight + (padTop+padBottom) < filtersHeight ||
      dataWidth + (padLeft+padRight) < filtersWidth) {
    mexErrMsgTxt("FILTERS are larger than the DATA (including padding).") ;
  }

  if (filtersHeight == 0 || filtersWidth == 0 || filtersDepth == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  packed_data_init_with_geom_int(&convIndices, false, convIndicesGeom, false, false, 0) ;

  conv_indices_cpu(convIndices.memoryInt, convIndices.geom.numElements,
    inMaskMode ? inIndices.memoryInt : NULL,
    maskMode ? maskIndices.memoryInt : NULL,
    maskMode ? maskIndicesLength : outputGeomHeight * outputGeomWidth,
    dataHeight, dataWidth, dataDepth,
    filtersHeight, filtersWidth,
    strideY, strideX,
    padTop, padBottom, padLeft, padRight);

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  if (inMaskMode) {
    packed_data_deinit(&inIndices);
  }
  if (maskMode) {
    packed_data_deinit(&maskIndices);
  }
  out[OUT_RESULT] = packed_data_deinit_extracting_array(&convIndices) ;
}
