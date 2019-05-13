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

#include <cassert>
#include <vector>

namespace exa {
  struct KdTreeNode {
    struct index_range {
      unsigned first;
      unsigned last;
    };

    __both__
    inline bool is_inner() const
    {
      return axis >> 30 != 3;
    }

    __both__
    inline bool is_leaf() const
    {
      return axis >> 30 == 3;
    }

    __both__
    inline int get_split() const
    {
      assert(is_inner());
      return split;
    }

    __both__
    inline int get_max_level() const
    {
      assert(is_inner());
      return max_level;
    }

    __both__
    inline unsigned get_axis()
    {
      assert(is_inner());
      return unsigned(axis >> 30);
    }

    __both__
    inline unsigned get_child(unsigned i = 0) const
    {
      assert(is_inner());
      return (first_child & 0x3FFFFFFF) + i;
    }

    __both__
    inline index_range get_indices() const
    {
      assert(is_leaf());
      return { first_prim, first_prim + (num_prims & 0x3FFFFFFF) };
    }

    __both__
    inline unsigned get_first_primitive() const
    {
      assert(is_leaf());
      return first_prim;
    }

    __both__
    inline unsigned get_num_primitives() const
    {
      assert(is_leaf());
      return num_prims & 0x3FFFFFFF;
    }

    __both__
    inline void set_inner(unsigned axis, int split, int max_level)
    {
      assert((int)axis >= 0 && axis < 3);
      this->axis = axis << 30;
      this->split = split;
      this->max_level = max_level;
    }

    __both__
    inline void set_leaf(unsigned first_primitive_index, unsigned count)
    {
      axis = 3ul << 30;
      first_prim = first_primitive_index;
      num_prims |= count;
    }

    __both__
    inline void set_first_child(unsigned index)
    {
      first_child |= index;
    }

  private:
    union
    {
      int split;
      unsigned first_prim;
    };

    union
    {
      // AA--.----.----.----
      unsigned axis;

      // --NP.NPNP.NPNP.NPNP
      unsigned num_prims;

      // --FC.FCFC.FCFC.FCFC
      unsigned first_child;
    };

    int max_level;
  };

  typedef std::vector<KdTreeNode> KdTree;

} // ::exa
