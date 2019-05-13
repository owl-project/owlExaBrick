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

#include <sstream>
#include <utility>

#include "ColorMapper.h"

static std::string trim(std::string str, std::string ws = " \t")
{
  // Remove leading whitespace
  auto first = str.find_first_not_of(ws);

  // Only whitespace found
  if (first == std::string::npos)
  {
    return "";
  }

  // Remove trailing whitespace
  auto last = str.find_last_not_of(ws);

  // No whitespace found
  if (last == std::string::npos)
  {
    last = str.size() - 1;
  }

  // Skip if empty
  if (first > last)
  {
    return "";
  }

  // Trim
  return str.substr(first, last - first + 1);
}

static std::vector<std::string> string_split(std::string s, char delim)
{
  std::vector<std::string> result;

  std::istringstream stream(s);

  for (std::string token; std::getline(stream, token, delim); )
  {
    result.push_back(token);
  }

  return result;
}

namespace exa {

  ColorMapper::ColorMapper(const vec3f* colors, std::size_t num)
    : values_(num)
  {
    for (std::size_t i=0; i<num; ++i)
    {
      values_[i] = { i/float(num-1), colors[i] };
    }
  }

  ColorMapper::ColorMapper(std::initializer_list<vec3f> colors)
    : values_(colors.size())
  {
    auto it = colors.begin();
    for (std::size_t i=0; i<colors.size(); ++i)
    {
      values_[i] = { i/float(colors.size()-1), *it++ };
    }
  }

  ColorMapper::ColorMapper(std::initializer_list<ControlPoint> colors)
    : values_(colors)
  {
  }

  ColorMapper::ColorMapper(std::string colorMapString)
  {
    // Not sure if delim='\n' works on every platform; tested on Linux
    std::vector<std::string> lines = string_split(colorMapString, '\n');

    std::vector<ControlPoint> controlPoints;
    for (std::string s : lines)
    {
      if (s.empty())
        continue;

      s = trim(s);

      ControlPoint cp;

      std::vector<std::string> reals = string_split(s, ',');

      if (reals.size() != 4) {
        std::cerr << "Not a valid control point: " << s << '\n';
        continue;
      }

      for (std::string& s : reals)
      {
        s = trim(s,"(");
        s = trim(s,")");
      }

      controlPoints.push_back({stof(reals[0]), {stof(reals[1]),stof(reals[2]),stof(reals[3])}});
    }

    auto cmpControlPoints = [](ControlPoint a, ControlPoint b) { return a.first < b.first; };
    if (!std::is_sorted(controlPoints.begin(),controlPoints.end(),cmpControlPoints))
    {
      std::cout << "Control points not sorted - sorting!\n";
      std::sort(controlPoints.begin(),controlPoints.end(),cmpControlPoints);
    }

    values_ = std::move(controlPoints);
  }

} // ::exa
