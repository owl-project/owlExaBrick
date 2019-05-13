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

// owl
#include "owl/common/math/box.h"

#ifdef __CUDACC__
#  define ON_DEVICE 1
#endif


namespace exa {

  using namespace owl;
  
  struct Brick {
    inline size_t numCells() const
    {
      return size.x*size_t(size.y)*size.z;
    }
    
    inline
#if ON_DEVICE
    __device__
#endif
    box3f getBounds() const
    {
      return box3f(vec3f(lower),
                   vec3f(lower + size*(1<<level)));
    }
    inline
#if ON_DEVICE
    __device__
#endif
    box3f getDomain() const
    {
      const float cellWidth = 1<<level;
      return box3f(vec3f(lower) - 0.5f*cellWidth,
                   vec3f(lower) + (vec3f(size)+0.5f)*cellWidth);
    }

    inline
#if ON_DEVICE
    __device__
#endif
    int getIndexIndex(const vec3i &idx) const
    {
      return begin + idx.x + size.x*(idx.y+size.y*idx.z);
    }
    vec3i    lower;
    vec3i    size;
    int      level;
    /*! offset into the scalar data index buffer (in which all bricks
        are stored sequentially) */
    uint32_t begin;
  };

} // ::exa
