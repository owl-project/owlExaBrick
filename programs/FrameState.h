// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cuda_runtime.h>
// gdt
#include "exa/common.h"
#include "owl/common/math/box.h"
#include "owl/common/math/AffineSpace.h"

namespace exa {

#define EMPTY_CELL_POISON_VALUE -1e20f
  
  struct FrameState {
    struct {
      vec3f pos;
      vec3f dir00;
      vec3f dirDu;
      vec3f dirDv;
    } camera;
    struct {
      bool  enabled { false };
      float value   {   0.f };
      int   channel { 0 };
    } isoSurface[MAX_ISO_SURFACES];
    struct {
      bool  enabled { false };
      vec3f normal { 1.f,0.f,0.f };
      int   channel { 0 };
      float offset { .5f };
    } contourPlane[MAX_CONTOUR_PLANES];
    struct {
      box3f coords;
      bool  enabled;
    } clipBox;
    
    struct {
      float length { 1e20f };
      bool  enabled { true };
    } ao;

    float clockScale { 0.f };
      
    
    affine3f voxelSpaceTransform;
    int   frameID;
    /*! domain over which the transfer function is defiend; anything
        out of this range will map to fully transparent black */
    interval<float> xfDomain[MAX_CHANNELS];
    cudaTextureObject_t xfTexture[MAX_CHANNELS];

    /*! the value we'll scale the final opacity value of each sample
      with - allows for making volume more transparent than a 0/1
      range for each sample would indicate */
    float xfOpacityScale { 1.f };
  };
}  
