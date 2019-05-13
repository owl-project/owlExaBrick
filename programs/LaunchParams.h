// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "../exa/KdTree.h"
#include "../exa/Regions.h"
#include "Brick.h"
#include "FrameState.h"

namespace exa {

  struct LaunchParams {
    //-------------------------------------------------------------------------
    // Render state
    //-------------------------------------------------------------------------

    int                    deviceIndex;
    int                    deviceCount;
    OptixTraversableHandle volumeBVH;
    OptixTraversableHandle surfaceModel;
    OptixTraversableHandle isoSurfaceBVH;
    OptixTraversableHandle streamlineBVH;
    vec2ui                 fbSize;
    vec3f                  worldSpaceBounds_lo;
    vec3f                  worldSpaceBounds_hi;
    vec3f                  voxelSpaceBounds_lo;
    vec3f                  voxelSpaceBounds_hi;
    uint32_t              *colorBufferPtr;
#ifdef __CUDA_ARCH__
    float4                *accumBufferPtr;
#else
    vec4f                 *accumBufferPtr;
#endif
    FrameState            *frameStateBuffer;
    float                  dt; // the rate of sampling relative to cell size
    SameBricksRegion      *sameBrickRegionsBuffer;
    int                   *sameBrickRegionsLeafList;
    Brick                 *brickBuffer;

    float                 *scalarBuffers;
    unsigned              *scalarBufferOffsets;

    // Fields for dvr rendering; 1st channel also affects isosurfaces
    int                   *primaryChannels;
    // Secondary field to use for colormapping the isosurface
    int                    colormapChannel;
    // Number of primary channels
    int                    numPrimaryChannels;

    // Direct volume rendering with gradient shading enabled/disabled
    int                    gradientShadingDVR;
    // Iso surface rendering with gradient shading enabled/disabled
    int                    gradientShadingISO;

    //-------------------------------------------------------------------------
    // tracer
    //-------------------------------------------------------------------------
    int                    tracerEnabled;
    vec3f                 *traces;
    vec3i                  tracerChannels;
    int                    numTraces;
    int                    numTimesteps;
    float                  steplen;
    int                   *currentTimestep; // buffer of one

  };

}
