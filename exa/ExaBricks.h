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

#include "common.h"

namespace exa {

  struct ExaBricks {
    typedef std::shared_ptr<ExaBricks> SP;
    
    struct Brick {
      typedef std::shared_ptr<Brick> SP;
      
      inline box3f getBounds() const;
      /*! get domain of all basis functions in this brick */
      inline box3f getDomain() const;

      vec3i size;
      vec3i lower;
      int   level;
      std::vector<int> cellIDs;
    };
    
    static ExaBricks::SP load(const std::string &brickFileName);

    box3f getBounds() const;
    
    /*! for stats and sanity-checking only: number of cells across all bricks... */
    size_t totalNumCells { 0 };
    std::vector<Brick::SP> bricks;
  };
  
  inline box3f ExaBricks::Brick::getBounds() const
  {
    return box3f(vec3f(lower),
                 vec3f(lower + size*(1<<level)));
  }

  /*! get domain of all basis functions in this brick */
  inline box3f ExaBricks::Brick::getDomain() const
  {
    const float cellWidth = 1<<level;
    return box3f(vec3f(lower) - 0.5f*cellWidth,
                 vec3f(lower) + (size+0.5f)*cellWidth);
  }

} // ::exa
