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

#include "exa/TriangleMesh.h"
#include "exa/ScalarField.h"
#include "exa/ExaBricks.h"
#include "exa/KdTree.h"

namespace exa {

  struct Config {
    typedef std::shared_ptr<Config> SP;
    

    void finalize();
    
    /*! return proper WORLD SPACE bounds, AFTER transformign voxel
        bounds back from voxel space to world space */
    box3f getBounds();

    static Config::SP parseConfigFile(const std::string &fileName);

    std::vector<TriangleMesh::SP> surfaces;
    struct {
      ExaBricks::SP    sp;
      box3f remap_from { vec3f(0.f), vec3f(1.f) };
      box3f remap_to   { vec3f(0.f), vec3f(1.f) };
      affine3f voxelSpaceTransform;
    } bricks;
    std::vector<ScalarField::SP> scalarFields;
  };
  
} // ::exa
