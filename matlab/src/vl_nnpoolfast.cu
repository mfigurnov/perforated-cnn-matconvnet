/** @file vl_nnpool.cu
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

#ifdef ENABLE_GPU
#include "bits/gpu.hpp"
#endif

#include <algorithm>
#include <assert.h>

/* option codes */
enum {
  opt_method = 0,
  opt_verbose
} ;

/* options */
vlmxOption  options [] = {
  {"Method",           1,   opt_method            },
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

VlEnumerator nnPoolMethodTypes [NN_POOL_METHODS_NUM] =
{
  {"Max",     (vl_index)NN_POOL_MAX     },
  {"Avg",     (vl_index)NN_POOL_AVG     },
} ;

enum {
  IN_DATA = 0, IN_INDICES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

#ifdef ENABLE_GPU

#endif

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  /* inputs */
  PackedData data ;
  PackedData indices ;
  PackedData derOutput ;

  /* outputs */
  PackedData output ;
  PackedData derData  ;
  PackedDataGeometry outputGeom ;
  PackedDataGeometry derDataGeom  ;

  PoolMethod method = NN_POOL_MAX;

  int poolSize;
  int outputHeight;
  int outputWidth;

#ifdef ENABLE_GPU
  bool gpuMode = false ;
#else
  bool const gpuMode = false ;
#endif
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;
  VlEnumerator *pair ;

  packed_data_init_empty(&data) ;
  packed_data_init_empty(&indices) ;
  packed_data_init_empty(&derOutput) ;
  packed_data_init_empty(&output) ;
  packed_data_init_empty(&derData) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  /* Throw an error if the input is not a GPU array. */
  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }
  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
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
      default: break ;
    }
  }

  packed_data_init_with_array(&data, in[IN_DATA]) ;
  packed_data_init_with_array_int(&indices, in[IN_INDICES]) ;
  if (backMode) { packed_data_init_with_array(&derOutput, in[IN_DEROUTPUT]) ; }

#if ENABLE_GPU
  gpuMode = (data.mode == matlabGpuArrayWrapper) ;
  if (gpuMode) {
    mxInitGPU() ;
  }
#endif

  /* check GPU/data class consistency */
  if (gpuMode && (derOutput.mode != matlabGpuArrayWrapper && backMode)) {
    mexErrMsgTxt("DATA is a GPU array but DEROUTPUT is not.") ;
  }
  if (! packed_data_are_compatible(&data, &indices)) {
    mexErrMsgTxt("DATA and INDICES are not both CPU or GPU arrays.") ;
  }
  if (data.geom.classID != mxSINGLE_CLASS) {
    mexErrMsgTxt("DATA is not of class SINGLE.");
  }
  if (indices.geom.classID != mxINT32_CLASS) {
    mexErrMsgTxt("INDICES is not of class INT32.");
  }
  if (backMode && (derOutput.geom.classID != mxSINGLE_CLASS)) {
    mexErrMsgTxt("DEROUTPUT is not of class SINGLE.");
  }

  if (gpuMode) {
    // indices.geom: [outputHeight, outputWidth, poolHeight * poolWidth, 1]
    outputHeight = indices.geom.height;
    outputWidth = indices.geom.width;
    poolSize = indices.geom.depth;
  } else {
    // indices.geom: [poolHeight * poolWidth, outputHeight, outputWidth, 1]
    poolSize = indices.geom.height;
    outputHeight = indices.geom.width;
    outputWidth = indices.geom.depth;
  }

  packed_data_geom_init(&outputGeom,
                        mxSINGLE_CLASS,
                        outputHeight,
                        outputWidth,
                        data.geom.depth,
                        data.geom.size) ;

  derDataGeom = data.geom ;

  if (verbosity > 0) {
    mexPrintf("vl_nnpoolfast: mode %s; %s\n", gpuMode?"gpu":"cpu", backMode?"backward":"forward") ;
    packed_data_geom_display(&data.geom, "vl_nnpoolfast: data") ;
    packed_data_geom_display(&indices.geom, "vl_nnpoolfast: indices") ;
    mexPrintf("vl_nnpoolfast: method: %s\n",
              vl_enumeration_get_by_value(nnPoolMethodTypes, method)->name);
    if (backMode) {
      packed_data_geom_display(&derOutput.geom, "vl_nnpoolfast: derOutput") ;
      packed_data_geom_display(&derDataGeom, "vl_nnpoolfast: derData") ;
    } else {
      packed_data_geom_display(&outputGeom, "vl_nnpoolfast: output") ;
    }
  }

  if (backMode) {
    if (derOutput.geom.height != outputGeom.height ||
        derOutput.geom.width != outputGeom.width ||
        derOutput.geom.depth != outputGeom.depth ||
        derOutput.geom.size != outputGeom.size)
    {
      mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (!backMode) {
    packed_data_init_with_geom(&output, gpuMode, outputGeom, false, false, 0) ;
  } else {
    packed_data_init_with_geom(&derData, gpuMode, derDataGeom, false, true, 0) ;
  }

  /* ---------------------------------------------------------- */
  /*                                               Forward mode */
  /* ---------------------------------------------------------- */
  if (backMode) {
    if (gpuMode) {
#ifdef ENABLE_GPU
      pooling_backward_gpu_fast<float>(derData.memory,
                                       data.memory,
                                       derOutput.memory,
                                       indices.memoryInt,
                                       method,
                                       data.geom.height * data.geom.width,
                                       data.geom.depth * data.geom.size,
                                       poolSize,
                                       derOutput.geom.height * derOutput.geom.width);
#endif
    } else {
      pooling_backward_cpu_fast<float>(derData.memory,
                                       data.memory,
                                       derOutput.memory,
                                       indices.memoryInt,
                                       method,
                                       data.geom.height * data.geom.width,
                                       data.geom.depth * data.geom.size,
                                       poolSize,
                                       derOutput.geom.height * derOutput.geom.width);
    }
  } else {
    if (gpuMode) {
#ifdef ENABLE_GPU
      pooling_gpu_fast<float>(output.memory,
                              data.memory,
                              indices.memoryInt,
                              method,
                              data.geom.height * data.geom.width,
                              data.geom.depth * data.geom.size,
                              poolSize,
                              output.geom.height * output.geom.width);
#endif
    } else {
      pooling_cpu_fast<float>(output.memory,
                              data.memory,
                              indices.memoryInt,
                              method,
                              data.geom.height * data.geom.width,
                              data.geom.depth * data.geom.size,
                              poolSize,
                              output.geom.height * output.geom.width);
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  packed_data_deinit(&data) ;
  packed_data_deinit(&indices);
  if (backMode) {
    packed_data_deinit(&derOutput) ;
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&derData) ;
  } else {
    out[OUT_RESULT] = packed_data_deinit_extracting_array(&output) ;
  }
}
