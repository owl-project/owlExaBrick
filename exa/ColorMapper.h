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

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "common.h"
#include "submodules/3rdParty/stb_image.h"

namespace exa {

  struct ColorMapper
  {
    typedef std::pair<float,vec3f> ControlPoint;

    /*! @param colors One color per interpolation point
        @param interpolation_points floats in [0..1], need _not_ be equally
        spaced but must be sorted!
        @param num number of colors and interpolation points */
    ColorMapper(const vec3f* colors, std::size_t num);

    ColorMapper(std::initializer_list<vec3f> colors);
    ColorMapper(std::initializer_list<ControlPoint> colors);

    /*! @param colorMapString A list of control points and colors to interpolate, e.g., "0.0,(0.0,0.1,0.2)\n1.0,(1.0,0.9,0.8)"
     */
    ColorMapper(std::string colorMapString);

    ColorMapper(const uint8_t *buf, std::size_t buf_size)
    {
        int w, h, n;
        uint8_t *img_data = stbi_load_from_memory(buf, buf_size, &w, &h, &n, 3);

        values_.reserve(w);
        for (std::size_t i = 0; i < w; ++i) {
            vec3f v;
            v.x = img_data[i * 3] / 255.f;
            v.y = img_data[i * 3 + 1] / 255.f;
            v.z = img_data[i * 3 + 2] / 255.f;
            values_.push_back({i/float(w-1), v});
        }

        stbi_image_free(img_data);
    }

    //! Reconstruct color at @param t
    vec3f operator()(float t) const {
        // This function assumes that the control points are sorted!

        // TODO: I'm sure this is defined _somewhere_ in gdt under a different name (?)
        auto lerp = [](const vec3f& a, const vec3f& b, float x) {  return (1.0f - x) * a + x * b; };

        auto it = std::upper_bound(values_.begin(),values_.end(),ControlPoint{t,{}},
                                   [](ControlPoint a, ControlPoint b) { return a.first < b. first; });

        std::size_t ival1 = it-values_.begin(); ival1 = ival1 > 0 ? ival1 - 1 : ival1;
        std::size_t ival2 = it-values_.begin(); ival2 = ival2 >= values_.size() ? ival2 - 1 : ival2;
        const float tval1 = values_[ival1].first;
        const float tval2 = values_[ival2].first;
        const float tval  = tval1==tval2 ? 1.f : (t-tval1)/(tval2-tval1);

        return lerp(values_[ival1].second, values_[ival2].second, tval);
    }

    std::vector<ControlPoint> values_;
  };

} // ::exa
