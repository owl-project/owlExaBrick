// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include "gdt/gdt.h"
#include "gdt/math/box.h"
#include "gdt/parallel/parallel_for.h"
// std
#include <vector>
#include <memory>
#include <mutex>
#include <fstream>
#include <map>

#define SAH 1

namespace exa {
  using gdt::vec3i;
  using gdt::vec4i;
  using gdt::box3i;
  using gdt::box4i;

  typedef int id_t;

  static const id_t invalid_id = (id_t)-1;
  
  int signedDivRoundDown(int a, int b)
  {
    if (a >= 0)
      return a/b;
    else
      return (a-(b-1))/b;
  }

  int signedDivRoundUp(int a, int b)
  {
    if (a >= 0)
      return (a+b-1)/b;
    else
      return a/b;
  }
  
  
  template<typename T>
  struct Array3 {
    Array3(const vec3i &size) { resize(size); }

    void fill(const T &init_t)
    { for (auto &t : elements) t = init_t; }
    
    void resize(const vec3i &size)
    {
      this->size = size;
      elements.resize(numElements());
    }
    
    size_t numElements() const { return size_t(size.x) * size_t(size.y) * size_t(size.z); }

    const typename std::vector<T>::const_iterator begin() const { return elements.begin(); }
    const typename std::vector<T>::const_iterator end() const { return elements.end(); }
    
    typename std::vector<T>::iterator begin() { return elements.begin(); }
    typename std::vector<T>::iterator end() { return elements.end(); }
    
    T &operator[](const vec3i &idx)
    {
      // PRINT(idx);
      // PRINT(size);
      assert(idx.x >= 0);
      assert(idx.x < size.x);
      assert(idx.y >= 0);
      assert(idx.y < size.y);
      assert(idx.z >= 0);
      assert(idx.z < size.z);
      size_t linearIdx = idx.x + size.x*(idx.y + size.y*(idx.z));
      assert(linearIdx < elements.size());
      return elements[linearIdx];
    }
    // const T &operator[](const vec3i &idx) const
    // {
    //   assert(idx.x >= 0);
    //   assert(idx.x < size.x);
    //   assert(idx.y >= 0);
    //   assert(idx.y < size.y);
    //   assert(idx.z >= 0);
    //   assert(idx.z < size.z);
    //   return elements[idx.x + size.x*(idx.y + size.y*(idx.z))];
    // }
    
    vec3i size;
    std::vector<T> elements;
  };
  
  struct SingleCell {
    inline box4i getBounds() const
    {
      // PRINT(lower);
      // PRINT(level);
      vec4i lo(lower,level);
      vec4i hi(lower+vec3i(1<<level),level+1);
      // PRINT(lo);
      // PRINT(hi);
      return box4i(lo,hi);
    }
    
    vec3i lower;
    int   level;
  };

  inline bool operator<(const SingleCell &a, const SingleCell &b)
  {
    return memcmp(&a,&b,sizeof(a)) < 0;
  }

  
  struct SingleCellModel {
    typedef std::shared_ptr<SingleCellModel> SP;

    inline box4i getBounds(id_t ID) const
    { assert(ID >= 0); assert(size_t(ID) < cells.size()); return cells[ID].getBounds(); }
      
    std::vector<SingleCell> cells;
  };

  namespace pp {
    box4i computeBounds(SingleCellModel::SP input, const std::vector<id_t> &IDs)
    {
      // PING;
      // PRINT(IDs.size());
      
      box4i result;
      std::mutex mutex;
      gdt::parallel_for_blocked(0,IDs.size(),4*1024,[&](size_t begin, size_t end){
          // PRINT(begin);
          // PRINT(end);
          box4i blockResult;
          for (int i=begin;i<end;i++) {
            blockResult.extend(input->getBounds(IDs[i]));
          }
          std::lock_guard<std::mutex> lock(mutex);
          result.extend(blockResult);
        });
      return result;
    }
  };

  struct Brick {
    typedef std::shared_ptr<Brick> SP;

    Brick(const box4i &bounds)
      : lower(vec3i(bounds.lower)),
        level(bounds.lower.w),
        cellIDs(vec3i(bounds.size()) / (1<<bounds.lower.w))
    {
    }

    size_t numInvalid() const
    {
      size_t count = 0;
      for (auto ID : cellIDs.elements)
        if (ID == invalid_id) count++;
      return count;
    }
    
    vec3i lower;
    int   level;
    Array3<id_t> cellIDs;
  };
  
  struct BrickedModel {
    typedef std::shared_ptr<BrickedModel> SP;

    void add(Brick::SP brick)
    {
      std::lock_guard<std::mutex> lock(mutex);
      bricks.push_back(brick);
    }

    std::mutex mutex;
    std::vector<Brick::SP> bricks;
  };

  std::vector<id_t> allIDsWithoutDuplicateCells(SingleCellModel::SP input)
  {
    std::map<SingleCell,id_t> mapping;
    for (size_t i=0;i<input->cells.size();i++)
      mapping[input->cells[i]] = i;
    std::vector<id_t> result;
    for (auto it : mapping)
      result.push_back(it.second);
    std::cout << "Created initial cell list - removed " << (input->cells.size()-result.size()) << " duplicated cells..." << std::endl;
    return result;
    
      // if (alreadyRead.find(cell) != alreadyRead.end()) {
      //   static size_t numDuplicates = 0;
      //   std::cout << "Duplicate: " << gdt::prettyNumber(++numDuplicates) << std::endl;
      //   continue;
      // }
      // alreadyRead.insert(cell);
  }   

  

  
  struct Bricker {
    static BrickedModel::SP buildBricks(SingleCellModel::SP input)
    {
      return Bricker(input).output;
    }
  private:
    Bricker(SingleCellModel::SP input)
      : input(input)
    {
      output = std::make_shared<BrickedModel>();
      std::vector<id_t> allIndices
        = allIDsWithoutDuplicateCells(input);
      buildRec(allIndices);
    }

    void buildRec(std::vector<id_t> &IDs)
    {
      // PING;
      // PRINT(IDs.size());
      // PRINT(IDs[0]);
      
      box4i bounds = pp::computeBounds(input,IDs);
      if (tryMakeLeaf(bounds,IDs)) return;

      int cellWidth = 1<<(bounds.upper.w-1);
#if SAH
      int bestDim = -1;
      int bestPos = -1;
      std::mutex mutex;
      float bestCost = std::numeric_limits<float>::infinity();
      gdt::parallel_for(3,[&](int dim){
      // for (int dim=0;dim<3;dim++) {
        int begin = signedDivRoundDown(bounds.lower[dim],cellWidth);
        int end   = signedDivRoundUp(bounds.upper[dim],(int)cellWidth);
        if ((end - begin) < 2) return;

        int numCand = std::min(7,end-begin-1);
        for (int candID=0;candID<numCand;candID++) {
          int cand = (begin+1)+size_t(candID*(end-begin-1))/numCand;
        // for (int cand=begin+1;cand<end;cand++) {
          int candPos = cand * cellWidth;
          box4i lBounds, rBounds;
          int lCount=0,rCount=0;
          for (auto &ID : IDs) {
            const SingleCell &c = input->cells[ID];
            if (c.lower[dim] >= candPos) {
              rBounds.extend(c.getBounds());
              rCount++;
            } else {
              lBounds.extend(c.getBounds());
              lCount++;
            }
          }
          if (lCount == 0 || rCount == 0)
            // not a valid split ...
            continue;

          float cost
            = lCount * lBounds.size().w * gdt::area(vec3i(lBounds.size()))
            + rCount * rBounds.size().w * gdt::area(vec3i(rBounds.size()));

          std::lock_guard<std::mutex> lock(mutex);
          if (cost >= bestCost) continue;
          bestCost = cost;
          bestDim = dim;
          bestPos = cand * cellWidth;
        }
      });
      if (bestDim < 0)
        throw std::runtime_error("could not find any sah split. This should not happen :-(.");
      int splitDim = bestDim;
      int splitPos = bestPos;
#else
      // width of _coarsest_ level cell
      int splitDim = arg_max(bounds.size());
      int begin = signedDivRoundDown(bounds.lower[splitDim],cellWidth);
      int end   = signedDivRoundUp(bounds.upper[splitDim],(int)cellWidth);
      if ((end - begin) < 2) {
        // PRINT(begin);
        // PRINT(end);
        // PRINT(cellWidth);
        // PRINT(bounds);
        // for (auto &ID : IDs) {
        //   const SingleCell &c = input->cells[ID];
        //   // PRINT(c.getBounds());
        // }
        throw std::runtime_error("no way to split the cells... this shouldn't happen!?");
      }

      // PRINT(begin);
      // PRINT(end);
      // PRINT(begin*cellWidth);
      // PRINT(end*cellWidth);
      int mid = (begin+end)/2;
      int splitPos = mid * cellWidth;
#endif
      std::vector<id_t> l, r;
      for (auto &ID : IDs) {
        const SingleCell &c = input->cells[ID];
        if (c.lower[splitDim] >= splitPos)
          r.push_back(ID);
        else
          l.push_back(ID);
      }

      if (l.empty() || r.empty()) {
        PING;
        PRINT(l.size());
        PRINT(r.size());
        PRINT(bounds);
        PRINT(splitDim);
        PRINT(splitPos);
        for (auto &ID : IDs) {
          const SingleCell &c = input->cells[ID];
          PRINT(c.getBounds());
        }

        throw std::runtime_error("invalid split...");
      };
      
      IDs.clear();
      gdt::parallel_for(2,[&](int side){
          buildRec(side?r:l);
        });
    }

    /*! check if this vector of IDs (with given bounds) can be made
        into a leaf, and if so, add that leaf and return true;
        otherwise return false */
    bool tryMakeLeaf(const box4i &bounds, std::vector<id_t> &IDs)
    {
      if (bounds.size().w > 1) return false;

      const size_t cellWidth = 1 << bounds.upper.w-1;
      const size_t cellVolume = cellWidth*cellWidth*cellWidth;

      // if we were to allow partially filled bricks we could remove
      // this:
      if (bounds.volume() != IDs.size()*cellVolume)
        return false;

      std::cout << "making leaf ... " << bounds.lower << " size " << (vec3i(bounds.size())/vec3i(cellWidth)) << " cw " << cellWidth << " cells " << IDs.size() << std::endl;
      
      Brick::SP brick = std::make_shared<Brick>(bounds);
      brick->cellIDs.fill(invalid_id);
      for (auto &ID : IDs) {
        assert(ID != invalid_id);
        SingleCell c = input->cells[ID];
        vec3i idx = (c.lower - vec3i(bounds.lower)) / int(cellWidth);
        // PRINT(c.lower);
        // PRINT(bounds);
        // PRINT(idx);
        // if (brick->cellIDs[idx] != invalid_id) {
        //   PING;
        //   PRINT(brick->cellIDs[idx]);
        //   PRINT(bounds);
        //   PRINT(c.lower);
        //   PRINT(cellWidth);
        //   PRINT(idx);
        // }
        
        // PRINT(ID);
        // PRINT(idx);
        brick->cellIDs[idx] = ID;
        // PRINT(brick->numInvalid());
      }
      for (auto ID : brick->cellIDs) {
        assert(ID != invalid_id);
      }

      output->add(brick);
      return true;
    }

    std::mutex outputMutex;
    SingleCellModel::SP input;
    BrickedModel::SP    output;
  };

  SingleCellModel::SP loadExaJet(const std::string &inFileName)
  {
    SingleCellModel::SP input = std::make_shared<SingleCellModel>();
    std::ifstream in(inFileName);
    while (!in.eof()) {
      SingleCell cell;
      in.read((char*)&cell,sizeof(cell));
      input->cells.push_back(cell);
    }
    std::cout << "done loading .... " << gdt::prettyNumber(input->cells.size()) << " cells" << std::endl;
    return input;
  }
  
  extern "C" int main(int ac, char **av)
  {
    assert(ac == 3);
    std::string inFileName = av[1];
    std::string outFileName = av[2];
    SingleCellModel::SP input = loadExaJet(inFileName);
    BrickedModel::SP output = Bricker::buildBricks(input);
    std::cout << "Done bricking, created " << output->bricks.size() << " bricks" << std::endl;

    std::ofstream out(outFileName);
    for (auto brick : output->bricks) {
      out.write((char *)&brick->lower,sizeof(brick->lower));
      out.write((char *)&brick->level,sizeof(brick->level));
      out.write((char *)brick->cellIDs.elements.data(),
                sizeof(id_t)*gdt::volume(brick->cellIDs.size));
    }
    
    return 0;
  }
  
};
