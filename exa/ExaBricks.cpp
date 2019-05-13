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

#include "ExaBricks.h"

namespace exa {

  ExaBricks::SP ExaBricks::load(const std::string &brickFileName)
  {
    std::cout << "#exa: loading exabricks from '" << brickFileName << "'" << std::endl;
    ExaBricks::SP exa = std::make_shared<ExaBricks>();
    std::ifstream in(brickFileName);
    if (!in.good()) throw std::runtime_error("could not open "+brickFileName);
    while (!in.eof()) {
      Brick::SP brick = std::make_shared<Brick>();
      in.read((char*)&brick->size,sizeof(brick->size));
      in.read((char*)&brick->lower,sizeof(brick->lower));
      in.read((char*)&brick->level,sizeof(brick->level));
      if (!in.good())
        break;
      brick->cellIDs.resize(owl::volume(brick->size));
      in.read((char*)brick->cellIDs.data(),brick->cellIDs.size()*sizeof(brick->cellIDs[0]));
      exa->totalNumCells += brick->cellIDs.size();
      exa->bricks.push_back(brick);
    }
    std::cout << "#exa: done loading exabricks, found "
              << owl::prettyDouble(exa->bricks.size()) << " bricks with "
              << owl::prettyDouble(exa->totalNumCells) << " cells" << std::endl;

    for (auto brick : exa->bricks) {
      for (auto cellID : brick->cellIDs) {
        assert(cellID >= 0
#if ALLOW_EMPTY_CELLS
               || cellID == -1
#endif
               );
        assert(cellID < exa->totalNumCells);
      }
    }
      
    return exa;
  }

  box3f ExaBricks::getBounds() const
  {
    box3f bounds;
    for (auto brick : bricks)
      bounds.extend(brick->getBounds());
    return bounds;
  }
    
} // ::exa
