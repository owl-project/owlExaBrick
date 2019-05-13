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

#include <cstddef>
#include "exa/ExaBricks.h"
// the device-definition of a brick
#include "programs/Brick.h"

namespace exa {

  /*! helper class that allows for keeping track which bricks overlap
      in which basis-function region */
  struct ExaBrickRegions {
    
    /*! denotes a region in which a given number of bricks overlap */
    struct BrickRegion {
      /* space covered by this region - should not overlap any other
         brickregion */
      box3f domain;
      /*! range of values of all cells overlapping this region */
      range1f valueRange;
      /*! offset in parent's leaflist class where our leaf list starts */
      int   leafListBegin;
      int   leafListSize;
      float finestLevelCellWidth;
    };

    void buildFrom(ExaBricks::SP exa,
                   // const std::vector<bool> &valid,
                   const exa::Brick *deviceBricks,
                   const size_t numBricks,
                   const float *scalarFields,
                   const unsigned *scalarFieldOffsets,
                   size_t numScalarFields);
    void addLeaf(std::vector<std::pair<box3f,int>> &buildPrims,
                 const box3f &domain);
    void buildRec(std::vector<std::pair<box3f,int>> &buildPrims,
                  const box3f &domain);
    void computeValueRange(BrickRegion &region,
                           ExaBricks::SP exa,
                           const exa::Brick *deviceBricks,
                           const float *scalarFields,
                           const unsigned *scalarFieldOffsets,
                           size_t numScalarFields);
    
    std::mutex mutex;
    std::vector<BrickRegion> brickRegions;
    /*! offset in parent's leaflist class where our leaf list starst */
    std::vector<int> leafList;
    ExaBricks::SP exa;
  };

  typedef ExaBrickRegions::BrickRegion SameBricksRegion;
  
} // ::exa
