// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "owl/common/parallel/parallel_for.h"
#include "owl/common/math/box.h"
#include "owl/common/math/AffineSpace.h"
// std
#include <fstream>
#include <deque>
#include <vector>
#include <mutex>
#include <set>

namespace exa {

  enum { BRICK_RAY_TYPE=0, ISO_RAY_TYPE, SURFACE_RAY_TYPE, SHADOW_RAY_TYPE, NUM_RAY_TYPES };
  using owl::vec2i;
  using owl::vec3f;
  using owl::linear3f;
  using owl::affine3f;
  using owl::interval;

  typedef owl::interval<float> range1f;
  
  static const int NUM_XF_VALUES = 128;

  static const int MAX_CHANNELS = 10;
  static const int MAX_ISO_SURFACES = 2;
  static const int MAX_CONTOUR_PLANES = 3;
  
  using owl::vec2i;
  using owl::vec3f;
  using owl::vec3i;
  using owl::vec4f;
  using owl::box3f;
  using owl::interval;
  using owl::affine3f;
  using owl::prettyNumber;
  using owl::prettyDouble;
  using owl::getCurrentTime;
  using owl::parallel_for;
  using owl::parallel_for_blocked;
  using owl::serial_for;
  using owl::serial_for_blocked;

  // from cmake:
#if BASIS_METHOD
# define EXPLICIT_BASIS_METHOD 1
#endif

} // ::exa
