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

#include <fstream>

#include "exa/Config.h"

namespace exa {

  void Config::finalize()
  {
    // ------------------------------------------------------------------
    // compute mapping from world space to voxel space (if
    // remap_from/remap_to were specified). If those values were not
    // specified this will return the unit transform
    // ------------------------------------------------------------------
    const box3f voxelSpaceBounds = bricks.remap_from;
    const box3f worldSpaceBounds = bricks.remap_to;

    affine3f voxelSpaceCoordSys
      = affine3f::translate(voxelSpaceBounds.lower)
      * affine3f::scale(voxelSpaceBounds.span());
      
    affine3f worldSpaceCoordSys
      = affine3f::translate(worldSpaceBounds.lower)
      * affine3f::scale(worldSpaceBounds.span());
      
    bricks.voxelSpaceTransform
      = voxelSpaceCoordSys
      * rcp(worldSpaceCoordSys);
  }

  /*! return proper WORLD SPACE bounds, AFTER transformign voxel
    bounds back from voxel space to world space */
  box3f Config::getBounds()
  {
    assert(bricks.sp);
    box3f bounds = bricks.sp->getBounds();
    bounds.lower = xfmPoint(rcp(bricks.voxelSpaceTransform),bounds.lower);
    bounds.upper = xfmPoint(rcp(bricks.voxelSpaceTransform),bounds.upper);
    return bounds;
  }

  Config::SP Config::parseConfigFile(const std::string &fileName)
  {
    Config::SP config = std::make_shared<Config>();

    std::cout << "opening config file "
              << fileName << std::endl;
    FILE *file = fopen(fileName.c_str(),"r");
      
    if (!file)
      throw std::runtime_error("error in opening config file '"+fileName+"'");
      
    static const int LINE_SZ = 10000;
    char line[LINE_SZ+1];
    // std::deque<std::string> tokens;
    std::vector<std::string> tokens;
    while (fgets(line,LINE_SZ,file) && !feof(file)) {
      char *tok = strtok(line," \t\n\r");
      while (tok && tok[0] != '#') {
        tokens.push_back(tok);
        tok = strtok(NULL," \t\n\r");
      }
    }
    fclose(file);

    // -------------------------------------------------------
    // now, parse the tokens
    // -------------------------------------------------------
    std::string basePath = fileName.substr(0,fileName.rfind('/'))+"/";
    for (std::vector<std::string>::const_iterator it = tokens.begin();
         it != tokens.end(); ) {

      if (*it == "remap_from") {
        config->bricks.remap_from.lower = vec3f(std::stof(it[+1]),
                                                std::stof(it[+2]),
                                                std::stof(it[+3]));
        config->bricks.remap_from.upper = vec3f(std::stof(it[+4]),
                                                std::stof(it[+5]),
                                                std::stof(it[+6]));
        it += 7;
        continue;
      }
        
      if (*it == "remap_to") {
        config->bricks.remap_to.lower = vec3f(std::stof(it[+1]),
                                              std::stof(it[+2]),
                                              std::stof(it[+3]));
        config->bricks.remap_to.upper = vec3f(std::stof(it[+4]),
                                              std::stof(it[+5]),
                                              std::stof(it[+6]));
        it += 7;
        continue;
      }
        
      if (*it == "scalar") {
        const std::string scalarName = it[+1];

        if (it[2] == "expr") {
          it += 3;
          std::vector<std::string> exprTokens;
          while (true) {
            exprTokens.push_back(*it);
            if (it->back() == '\"')
              break;
            it++;
          }
          it += 1;
          config->scalarFields.push_back(ScalarField::createFromExpression(scalarName,config->scalarFields,exprTokens));
        } else {
          const std::string fileName = basePath+it[2];
          it += 3;
          config->scalarFields.push_back(ScalarField::load(scalarName,fileName));
        }

        continue;
      }
        
      if (*it == "vector") {
        const std::string fieldName = it[+1];
        const std::string fnx = basePath+it[2];
        const std::string fny = basePath+it[3];
        const std::string fnz = basePath+it[4];
        it += 5;
        config->scalarFields.push_back(ScalarField::loadAndComputeMagnitude(fieldName,fnx,fny,fnz));

        continue;
      }
        
      if (*it == "value_range") {
        float lo = std::stof(it[+1]);
        float hi = std::stof(it[+2]);
        it += 3;
          
        assert(!config->scalarFields.empty());
        config->scalarFields.back()->valueRange.lower = lo;
        config->scalarFields.back()->valueRange.upper = hi;

        continue;
      }
        

      if (*it == "bricks") {
        std::string fileName = basePath+it[+1];
        it += 2;
        std::cout << "#exa: loading bricks from " << fileName << std::endl;
        config->bricks.sp = ExaBricks::load(fileName);
        continue;
      }

      if (*it == "triangles") {
        std::string fileName = basePath+it[+1];
        it += 2;
        std::cout << "#exa: loading triangles from " << fileName << std::endl;
        config->surfaces = TriangleMesh::load(fileName);
        continue;
      }

      throw std::runtime_error("error in parsing config file: unknown token '"+*it+"'");
    }


    config->finalize();
      
    return config;
  }
  
} // ::exa
