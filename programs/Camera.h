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

#include "owl/common/math/random.h"
#include "FrameState.h"

namespace exa {
   
 typedef owl::common::LCG<16> Random;

  struct Camera {
    static __device__ owl::Ray generateRay(const FrameState &fs,
                                           const vec2f &pixelSample,
                                           Random &rnd) 
    {
      // const vec3f rd = 0.f; //camera_lens_radius * random_in_unit_disk(rnd);
      // const vec3f lens_offset = fs->camera_u * rd.x + fs->camera_v * rd.y;
      const vec3f origin = fs.camera.pos;// + lens_offset;
      const vec3f direction
        = fs.camera.dir00
        + pixelSample.x * fs.camera.dirDu
        + pixelSample.y * fs.camera.dirDv
        ;
  
      return owl::Ray(/* origin   : */ origin,
                      /* direction: */ normalize(direction),
                      /* tmin     : */ 1e-6f,
                      /* tmax     : */ 1e8f);
    }
  };
  
}

