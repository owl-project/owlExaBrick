// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "owl/owl.h"
#include "owl/common/math/box.h"
#include "owl/common/parallel/parallel_for.h"
// exa
#include "../exa/KdTree.h"
// std
#include <vector>
#include <memory>
#include <mutex>
#include <fstream>
#include <map>
#include <set>
#include <atomic>

#define SPATIAL_MEDIAN_BUILDER    0
#define SAH_ALIKE_BUILDER         1
#define SMALL_BRICK_COUNT_BUILDER 2

namespace exa {
  using owl::vec3d;
  using owl::vec3i;
  using owl::vec4i;
  using owl::box3i;
  using owl::box4i;

  bool verbose = false;
  bool parallel = false;
  
  typedef int id_t;

  static const id_t invalid_id = (id_t)-1;

  /*! maximum leaf width, in any dimension */
  int maxLeafWidth = 127;
  // static const int maxLeafWidth = 255;
  
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
    uint64_t *pa = (uint64_t *)&a;
    uint64_t *pb = (uint64_t *)&b;
    if (pa[0] < pb[0]) return true;
    if (pa[0] > pb[0]) return false;
    return pa[1] < pb[1];
  }
  inline bool operator!=(const SingleCell &a, const SingleCell &b)
  {
    return a.lower != b.lower || a.level != b.level;
  }

  
  struct SingleCellModel {
    typedef std::shared_ptr<SingleCellModel> SP;

    inline box4i getBounds(id_t ID) const
    {
      assert(ID >= 0);
      assert(size_t(ID) < cells.size());
      return cells[ID].getBounds();
    }
      
    std::vector<SingleCell> cells;
  };


  size_t unitCellVolume(const box4i &box)
  {
    const vec4i size = box.size();
    return size.x*size_t(size.y)*size_t(size.z);
  }

  size_t area(const box4i &box)
  {
    const vec4i size = box.size();
    return
      size.x*size_t(size.y) +
      size.y*size_t(size.z) +
      size.z*size_t(size.x);
  }
  
  
  box4i computeBounds(size_t &volumeOccupied,
                      SingleCellModel::SP input,
                      const std::vector<id_t> &IDs)
  {
    box4i result;
    std::mutex mutex;
    owl::serial_for_blocked(0,IDs.size(),4*1024,[&](size_t begin, size_t end){
        box4i blockResult;
        size_t blockVolumeOccupied = 0;
        for (size_t i=begin;i<end;i++) {
          const box4i bounds_i = input->getBounds(IDs[i]);
          blockResult.extend(bounds_i);
          blockVolumeOccupied += unitCellVolume(bounds_i);
        }
        std::lock_guard<std::mutex> lock(mutex);
        result.extend(blockResult);
        volumeOccupied += blockVolumeOccupied;
      });
    return result;
  }

  box4i computeCoarsestLevelBounds(SingleCellModel::SP input,
                                   const std::vector<id_t> &IDs)
  {
    size_t volumeOccupied = 0;
    box4i bounds = computeBounds(volumeOccupied, input, IDs);
    const int cellWidth = 1 << (bounds.upper.w - 1);

    if (verbose)
      std::cout << "splitting " << owl::prettyNumber(IDs.size())
              << " bounds " << bounds << std::endl;
    
    for (int d=0;d<3;d++) {
      bounds.lower[d] = cellWidth * signedDivRoundDown(bounds.lower[d],cellWidth);
      bounds.upper[d] = cellWidth * signedDivRoundUp(bounds.upper[d],cellWidth);
    }

    if (verbose) {
    std::cout << "coarse bounds " << bounds << std::endl;
    PRINT(volumeOccupied);
    PRINT(unitCellVolume(bounds));
    std::cout << "occupied = " << int(volumeOccupied * 100. / unitCellVolume(bounds))
              << "%" << std::endl;
    }
    return bounds;
  }
  
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

  int hash(const SingleCell &c)
  {
    int v = c.lower.x;
    v = v*13 ^ 0x23123;
    v = v*7  ^ 0x3898;
    v ^= (v>>5);
    v ^= (v<<7);
    v = v & c.lower.y;
    v = v*13 ^ 0x23123;
    v = v*7  ^ 0x3898;
    v ^= (v>>5);
    v ^= (v<<7);
    v = v & c.lower.z;
    v = v*13 ^ 0x23123;
    v = v*7  ^ 0x3898;
    v ^= (v>>5);
    v ^= (v<<7);
    v = v & c.level;
    v = v*13 ^ 0x23123;
    v = v*7  ^ 0x3898;
    v ^= (v>>5);
    v ^= (v<<7);
    return v;
  }
  struct FindDuplicatesHashEntry {
    std::mutex mutex;
    std::map<SingleCell,int> knownCells;

    bool isDuplicate(const SingleCell &c)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (knownCells.find(c) != knownCells.end()) return true;
      knownCells[c] = 1;
      // knownCells.insert(c);
      return false;
    }
  };
  
  std::vector<id_t> allIDsWithoutDuplicateCells(SingleCellModel::SP input)
  {
#if 1
    std::cout << "creation of initial, duplicate-free prim list (via sort)" << std::endl;
    std::vector<std::pair<SingleCell,id_t>> pairedVector(input->cells.size());
    owl::serial_for_blocked(0ULL,input->cells.size(),16*1024,[&](size_t begin, size_t end){
        for (size_t i=begin;i<end;i++) 
          pairedVector[i] = { input->cells[i], (id_t)i };
      });
    std::cout << " ... unsorted done" << std::endl;
    std::sort(pairedVector.begin(),pairedVector.end());
    std::cout << " ... sort done" << std::endl;
    std::vector<id_t> result;
#if 1
    // first: overwrite inner nodes: vector is already sorted, so for
    // every inner node there should be a leaf node with same pos
    // "somewhare" around it... so first, do a pass where for every
    // element we check if the preceding elements have same position,
    // and if so, overwrite them with finer level if possible. that'll
    // still leave duplicates, but those get removed later below
    for (int i=1;i<pairedVector.size();i++) {
      for (int j=i-1;
           j>=0
             &&
             pairedVector[j].first.lower == pairedVector[i].first.lower
             ;--j) {
        if (pairedVector[j].first.level > pairedVector[i].first.level)
          pairedVector[j] = pairedVector[i];
      }
      
      int j = i-1;
      if (pairedVector[j].first.lower != pairedVector[j].first.lower)
        continue;
      for (int j=i-1;j>=0;--j) {
      }
    }
    
    result.push_back(pairedVector[0].second);
    for (int i=1;i<pairedVector.size();i++) {
      if (pairedVector[i].first != pairedVector[i-1].first)
        result.push_back(pairedVector[i].second);
    }
#else
    result.push_back(pairedVector[0].second);
    for (int i=1;i<pairedVector.size();i++) {
      if (pairedVector[i].first != pairedVector[i-1].first)
        result.push_back(pairedVector[i].second);
    }
#endif
    return result;

#elif 1
    std::cout << "(Serial) creation of initial, duplicate-free prim list" << std::endl;
    std::mutex mutex;
    std::vector<id_t> result;
    const int hashBuckets = 256; // enough for 256-ish threads ...
    std::vector<FindDuplicatesHashEntry *> hashBucket(hashBuckets);
    owl::serial_for(hashBucket.size(),[&](size_t idx){
        // we _intentionally_ use new/delete here to spread the
        // create/delete cost over multiple threads (std::set and
        // set::map are very expensive to free)
        hashBucket[idx] = new FindDuplicatesHashEntry;
      });

    owl::serial_for_blocked(0ULL,input->cells.size(),16*1024,[&](size_t begin, size_t end){
        std::vector<id_t> blockResult;
        for (size_t i=begin;i<end;i++) {
          const SingleCell &c = input->cells[i];
          if (hashBucket[hash(c) % hashBuckets]->isDuplicate(c))
            continue;
          blockResult.push_back(i);
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (auto ID : blockResult)
          result.push_back(ID);
      });

    owl::serial_for(hashBucket.size(),[&](size_t idx){
        // we _intentionally_ use new/delete here to spread the
        // create/delete cost over multiple threads (std::set and
        // set::map are very expensive to free)
        delete hashBucket[idx];
      });
    std::cout << "Created initial cell list - removed " << (input->cells.size()-result.size()) << " duplicated cells..." << std::endl;
    return result;
        
#else
    std::map<SingleCell,id_t> mapping;
    std::vector<id_t> result;
    for (auto it : mapping)
      result.push_back(it.second);
    std::cout << "Created initial cell list - removed " << (input->cells.size()-result.size()) << " duplicated cells..." << std::endl;
    return result;
#endif
    // if (alreadyRead.find(cell) != alreadyRead.end()) {
    //   static size_t numDuplicates = 0;
    //   std::cout << "Duplicate: " << owl::prettyNumber(++numDuplicates) << std::endl;
    //   continue;
    // }
    // alreadyRead.insert(cell);
  }   

  

  
  struct Bricker {
    static BrickedModel::SP buildBricks(SingleCellModel::SP input, int builderType=SPATIAL_MEDIAN_BUILDER)
    {
      return Bricker(input,builderType).output;
    }

    Bricker(SingleCellModel::SP input, int builderType=SPATIAL_MEDIAN_BUILDER)
      : input(input)
    {
      output = std::make_shared<BrickedModel>();

      // Create root node
      tree.emplace_back();

      std::vector<id_t> allIndices
        = allIDsWithoutDuplicateCells(input);
      switch (builderType) {
      case SPATIAL_MEDIAN_BUILDER:
        buildRec<SPATIAL_MEDIAN_BUILDER>(allIndices, 0);
        break;
      case SAH_ALIKE_BUILDER:
        buildRec<SAH_ALIKE_BUILDER>(allIndices, 0);
        break;
      case SMALL_BRICK_COUNT_BUILDER:
        buildRec<SMALL_BRICK_COUNT_BUILDER>(allIndices, 0);
        break;
      }
    }

    template<int BuilderType>
    void buildRec(std::vector<id_t>& IDs, unsigned nodeID);


    size_t reduce_mul(const vec3i &v)
    {
      return v.x*size_t(v.y)*v.z;
    }
       
    /*! check if this vector of IDs (with given bounds) can be made
      into a leaf, and if so, add that leaf and return true;
      otherwise return false */
    bool tryMakeLeaf(box4i bounds, std::vector<id_t> &IDs, unsigned nodeID)
    {
      if (IDs.size() == 0)
        throw "empty ID vector in tryMakeLeaf!?";
    
      if (bounds.size().w > 1) return false;
      
      const size_t cellWidth = 1ULL << (bounds.upper.w-1);

      if (verbose) {
        std::cout << "......................................................." << std::endl;
        PING; PRINT(bounds);
        PRINT(bounds.size());
        PRINT(cellWidth);
      }
      if (bounds.size().x/cellWidth > maxLeafWidth) return false;
      if (bounds.size().y/cellWidth > maxLeafWidth) return false;
      if (bounds.size().z/cellWidth > maxLeafWidth) return false;
      
      const size_t cellVolume = cellWidth*cellWidth*cellWidth;
      
      if (verbose) {
        PRINT(bounds.volume());
        PRINT(IDs.size());
      }
      
#if ALLOW_EMPTY_CELLS
      // ignore this test in this mode ...
#else
      // if we were to allow partially filled bricks we could remove
      // this:
      if (bounds.volume() != IDs.size()*cellVolume)
        return false;
#endif


#if ALLOW_EMPTY_CELLS
      // in empty-cells mode we can't guarantee that all empty space
      // was cut off before this code is called, so make sure to
      // rebuild the brick bounds first
      bounds = box4i();
      for (auto &ID : IDs) {
        assert(ID != invalid_id);
        SingleCell c = input->cells[ID];
        bounds.extend(c.getBounds());
      }
#endif



      
      static std::atomic<int> leafCounter(0);
#if 1
      int leafID = leafCounter++;
      static size_t numCellsInLeaves = 0;
      numCellsInLeaves += IDs.size(); //reduce_mul(vec3i(bounds.size())/vec3i(cellWidth));

      {
        static std::mutex outputMutex;
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cout << "making leaf #" << leafID << "... " << bounds.lower << " size " << (vec3i(bounds.size())/vec3i(cellWidth)) << " cw " << cellWidth << " cells " << IDs.size() << ", avg leaf size " << (numCellsInLeaves/float(leafID+1)) << std::endl;
      }
#endif


      
      Brick::SP brick = std::make_shared<Brick>(bounds);
      brick->cellIDs.fill(invalid_id);
      for (auto &ID : IDs) {
        assert(ID != invalid_id);
        SingleCell c = input->cells[ID];
        vec3i idx = (c.lower - vec3i(bounds.lower)) / int(cellWidth);
        brick->cellIDs[idx] = ID;
        assert(ID >= 0);
        assert(ID < input->cells.size());
      }
      for (auto ID : brick->cellIDs) {
        assert(ID != invalid_id);
      }

      tree[nodeID].set_leaf((unsigned)output->bricks.size(), 1U);
      output->add(brick);
      return true;
    }

    std::mutex outputMutex;
    SingleCellModel::SP input;
    BrickedModel::SP    output;
    KdTree              tree;
  };

  template<int BuilderType>
  void Bricker::buildRec(std::vector<id_t>& IDs, unsigned nodeID)
  {
    const box4i coarseBounds
      = computeCoarsestLevelBounds(input, IDs);
    if (tryMakeLeaf(coarseBounds, IDs, nodeID)) return;

    const int coarseCellWidth = 1 << (coarseBounds.upper.w - 1);

    // ------------------------------------------------------------------
    // create one bin per coarse cell, in each dimension; for each of
    // those "slices" of the volume, track total cell volume used (to
    // figure out if it's completely full), as well as its bounds (to
    // track if its all coarse, or also contains finer levels
    // ------------------------------------------------------------------
    const vec3i coarseGridDims = (vec3i(coarseBounds.size()) / vec3i(coarseCellWidth));
    if (coarseGridDims == vec3i(1)) {
      throw std::runtime_error("coarse size 1 that's not a leaf!?");
    }
    struct {
      std::vector<size_t> volumeUsed;
      std::vector<std::vector<int>>    levels;
      std::vector<box4i>  bounds;
    } dim[3];

    // first, allocate and initialze each of these slices
    for (int d=0;d<3;d++) {
      dim[d].bounds.resize(coarseGridDims[d]);
      dim[d].levels.resize(coarseGridDims[d]);
      dim[d].volumeUsed.resize(coarseGridDims[d]);
      for (int i=0;i<coarseGridDims[d];i++)
        dim[d].volumeUsed[i] = 0;
      for (int i=0;i<coarseGridDims[d];i++)
        dim[d].bounds[i] = box4i();
    }

    // second, 'splat' all cells into that
    owl::parallel_for(3,[&](int d) {
        for (auto ID : IDs) {
          const SingleCell& cell = input->cells[ID];
          const box4i cellBounds = cell.getBounds();
          const vec3i cellBins
            = vec3i(cellBounds.lower-coarseBounds.lower)
            / coarseCellWidth;
          // for (int d=0;d<3;d++) {
          dim[d].volumeUsed[cellBins[d]] += unitCellVolume(cellBounds);
          dim[d].bounds[cellBins[d]].extend(cellBounds);
          if (std::find(dim[d].levels[cellBins[d]].begin(),
                        dim[d].levels[cellBins[d]].end(),
                        cell.level) == dim[d].levels[cellBins[d]].end())
            dim[d].levels[cellBins[d]].push_back(cell.level);
        }
      });
    
    int maxLevel = -1;
    for (int d=0;d<3;d++) {
      for (int slice=0;slice<coarseGridDims[d];slice++) {
        for (size_t l=0;l<dim[d].levels[slice].size();++l)
          maxLevel = std::max(maxLevel,dim[d].levels[slice][l]);
      }
    }

    // ------------------------------------------------------------------
    // now that we have all that info, find the best split pos
    // ------------------------------------------------------------------
    int bestSplitDim = -1;
    int bestSplitPos = -1;
    double bestSplitCost = std::numeric_limits<float>::infinity();//1ull << 62;
    if (BuilderType == SAH_ALIKE_BUILDER || BuilderType == SMALL_BRICK_COUNT_BUILDER) {
      for (int d=0;d<3;d++) {
        if (coarseGridDims[d] == 0) continue;
        // the (unit cell) volume of a given slice in this dimension,
        // _if_ it was full, no matter which cells it is filled with
        const size_t expectedVolumeOfSlice
          = unitCellVolume(coarseBounds) / coarseGridDims[d];
        for (int planeID=1;planeID<coarseGridDims[d];planeID++) {
          const int leftSlice  = planeID-1;
          const int rightSlice = planeID;
        
          // now, look at slices just left and right of this plane
          const bool sliceToLeftIsFull
            = dim[d].volumeUsed[leftSlice ] == expectedVolumeOfSlice;
          const bool sliceToRightIsFull
            = dim[d].volumeUsed[rightSlice] == expectedVolumeOfSlice;

          const bool sliceToLeftIsAllCoarse
            = dim[d].bounds[leftSlice ].upper.w == coarseBounds.upper.w;
          const bool sliceToRightIsAllCoarse
            = dim[d].bounds[rightSlice].upper.w == coarseBounds.upper.w;

          const bool sliceToLeftHasSomeFine = !sliceToLeftIsAllCoarse;
          const bool sliceToRightHasSomeFine= !sliceToRightIsAllCoarse;
          const bool sliceToLeftHasSomeCoarse 
            = dim[d].bounds[leftSlice ].lower.w == coarseBounds.lower.w;
          const bool sliceToRightHasSomeCoarse 
            = dim[d].bounds[rightSlice].lower.w == coarseBounds.lower.w;

#if 1
          const int llo = dim[d].bounds[leftSlice].lower.w;
          const int lsz = dim[d].bounds[leftSlice].size().w;
          const int rlo = dim[d].bounds[rightSlice].lower.w;
          const int rsz = dim[d].bounds[rightSlice].size().w;
          const bool lFull = sliceToLeftIsFull;
          const bool rFull = sliceToRightIsFull;
          const bool isBoundary
            = !(llo == rlo && lsz == rsz && lFull && rFull);
#else
          // check if this is a boundary slice...
          const bool isBoundary
            =  (sliceToLeftIsAllCoarse && !sliceToRightIsAllCoarse)
            || (sliceToRightIsAllCoarse && !sliceToLeftIsAllCoarse)
            || (sliceToLeftIsFull && !sliceToRightIsFull)
            || (sliceToRightIsFull && !sliceToLeftIsFull)
            || (sliceToLeftHasSomeCoarse && !sliceToRightHasSomeCoarse)
            || (!sliceToLeftHasSomeCoarse && sliceToRightHasSomeCoarse)
            ;
#endif
          // if _not_ a boundary - ie, we'd be slicing at a place that's
          // equally good or bad on both left and right - just ignore it
          if (!isBoundary)
            continue;

          // ------------------------------------------------------------------
          // OK, it's a boundary, so a potential split candidate:
          // ------------------------------------------------------------------
          // compute bounding boxes of _all_ slices left and right of
          // this plane:
          box4i leftBounds;
          std::vector<int> leftLevels;
          for (int slice=0;slice<planeID;slice++) {
            leftBounds.extend(dim[d].bounds[slice]);
            for (size_t l=0;l<dim[d].levels[slice].size();++l)
              if (std::find(leftLevels.begin(),leftLevels.end(),
                            dim[d].levels[slice][l]) == leftLevels.end())
                leftLevels.push_back(dim[d].levels[slice][l]);
          }
          box4i rightBounds;
          std::vector<int> rightLevels;
          for (int slice=planeID;slice<coarseGridDims[d];slice++) {
            rightBounds.extend(dim[d].bounds[slice]);
            for (size_t l=0;l<dim[d].levels[slice].size();++l)
              if (std::find(rightLevels.begin(),rightLevels.end(),
                            dim[d].levels[slice][l]) == rightLevels.end())
                rightLevels.push_back(dim[d].levels[slice][l]);
          }
        
#if 0
          const size_t splitPlaneArea
            = coarseBounds.size()[(d+1)%3] * coarseBounds.size()[(d+2)%3];
          const double cost
            = splitPlaneArea
            * (1+fabsf(planeID-0.5f*coarseGridDims[d])///float(coarseCellWidth)
               )
            * std::min(leftBounds.size().w,rightBounds.size().w);
          ;
#else
          double cost = 0.;
          if (BuilderType == SAH_ALIKE_BUILDER)
            cost = area(leftBounds) * (double)unitCellVolume(leftBounds) * leftBounds.size().w
            + area(rightBounds) * (double)unitCellVolume(rightBounds) * rightBounds.size().w;
          else
            cost = (double)leftLevels.size() + (double)rightLevels.size();
#endif
        
          if (verbose) {
            std::cout << " split " << planeID << "@" << d << ": " << leftBounds << "+" << rightBounds << " cost " << cost << std::endl;
          }
          if (cost < bestSplitCost) {
            bestSplitCost = cost;
            bestSplitDim = d;
            bestSplitPos = coarseBounds.lower[d] + planeID * coarseCellWidth;
          } else if (BuilderType == SMALL_BRICK_COUNT_BUILDER && cost == bestSplitCost) {
            // That's what Kaehler does: in case of ambiguity, prefer splits
            // that are closer to the middle/spatial median position.
            // That might also be a good idea to do in the "sah-alike" case (?)
            const int middlePos = coarseGridDims[bestSplitDim] / 2;
            const int thisSplitPos = coarseBounds.lower[d] + planeID * coarseCellWidth;
            if (std::abs(thisSplitPos-middlePos)<std::abs(bestSplitPos-middlePos)) {
              bestSplitCost = cost;
              bestSplitDim = d;
              bestSplitPos = thisSplitPos;
            }
          }
        }
      }
    } else
      assert(BuilderType == SPATIAL_MEDIAN_BUILDER && "Oups");

    // fallback: if no boundary plane could be found at all, simply
    // split in the middle
    if (bestSplitDim == -1) {
      if (verbose) {
      std::cout << "no boundary split - split spatial median!" << std::endl;
      }
      bestSplitDim = arg_max(coarseGridDims);
      int planeID = coarseGridDims[bestSplitDim] / 2;
      bestSplitPos = coarseBounds.lower[bestSplitDim] + planeID * coarseCellWidth;
    } else
      std::cout << "DID find a split ..." << std::endl;
    
    if (1) {
      if (verbose) {
      std::cout << "==================================================================" << std::endl;
      std::cout << "splitting " << owl::prettyNumber(IDs.size())
                << " cells" << std::endl;
      std::cout << " - coarse bounds : " << coarseBounds
                << " dims " << coarseGridDims << std::endl;
      std::cout << "best dim " << bestSplitDim << " pos " << bestSplitPos << std::endl;
      }
    }

    // ==================================================================
    // perform the actual partition
    // ==================================================================
      
    std::vector<id_t> l, r;
    box4i actual_lBounds, actual_rBounds;
    for (auto ID : IDs) {
      const SingleCell &cell       = input->cells[ID];
      const box4i       cellBounds = cell.getBounds();
      if (cellBounds.lower[bestSplitDim] >= bestSplitPos) {
        r.push_back(ID);
        actual_rBounds.extend(cellBounds);
      } else if (cellBounds.upper[bestSplitDim] <= bestSplitPos) {
        l.push_back(ID);
        actual_lBounds.extend(cellBounds);
      } else {
        PRINT(cellBounds);
        PRINT(bestSplitDim);
        PRINT(bestSplitPos);
        throw std::runtime_error("cell straddles split plane!?");
      }
    }

    if (verbose) {
      std::cout << "got left  " << owl::prettyNumber(l.size())
                << " " << actual_lBounds << std::endl;
      std::cout << "got right " << owl::prettyNumber(r.size())
                << " " << actual_rBounds << std::endl;
    }
    
      if (l.empty() || r.empty()) {
        PING;
        PRINT(l.size());
        PRINT(r.size());
        PRINT(coarseBounds);
        PRINT(bestSplitDim);
        PRINT(bestSplitPos);
        for (auto& ID : IDs) {
          const SingleCell& c = input->cells[ID];
          PRINT(c.getBounds());
        }
      
        throw std::runtime_error("invalid split...");
      };

      unsigned firstChildIndex = (unsigned)tree.size();

      tree.emplace_back();
      tree.emplace_back();

      tree[nodeID].set_inner((unsigned)bestSplitDim,bestSplitPos,maxLevel);
      tree[nodeID].set_first_child(firstChildIndex);

      IDs.clear();
      if (parallel)
        owl::parallel_for(2, [&](int side) {
            buildRec<BuilderType>(side ? r : l, firstChildIndex+side);
          });
      else
        owl::serial_for(2, [&](int side) {
            buildRec<BuilderType>(side ? r : l, firstChildIndex+side);
          });
    }
  
    SingleCellModel::SP loadExaJet(const std::string &inFileName)
    {
      std::cout << "trying to load exajet single-cell file format from file " << inFileName << std::endl;
      SingleCellModel::SP input = std::make_shared<SingleCellModel>();
      std::ifstream in(inFileName);
      owl::interval<int> levelRange;
      size_t ping = 1;
      while (!in.eof()) {
        SingleCell cell;
        in.read((char*)&cell,sizeof(cell));
        if (!in.good()) break;
        input->cells.push_back(cell);
        levelRange.extend(cell.level);
        if (input->cells.size() >= ping) {
          std::cout << " .... got " << owl::prettyNumber(input->cells.size()) << " cells, levels = " << levelRange << std::endl;
          ping += ping;
        }
      }
      std::cout << "done loading .... got " << owl::prettyNumber(input->cells.size()) << " cells ... " <<  std::endl;
      std::cout << "range of levels: " << levelRange << std::endl;
      return input;
    }
  
    extern "C" int main(int argc, char **argv)
    {
      try {
        bool spatialMedianBuilder = false;
        bool largeBricks = false;
        std::string inFileName;
        std::string outFileName;
        std::string kdFileName; // optional
        for (int i = 1; i < argc; i++) {
          const std::string arg = argv[i];
          if (arg[0] != '-')
            inFileName = arg;
          else if (arg == "-o")
            outFileName = argv[++i];
          else if (arg == "-kd")
            kdFileName = argv[++i];
          else if (arg == "--parallel")
            parallel = true;
          else if (arg == "--max-leaf-width")
            maxLeafWidth = std::stoi(argv[++i]);
          else if (arg == "-v")
            verbose = true;
          else if (arg == "--no-shift-planes" || arg == "--no-planes" ||
                   arg == "--spatial-median" || arg == "--spatial-median-builder")
            spatialMedianBuilder = true;
          else if (arg == "--large-bricks")
            largeBricks = true;
          else
            throw std::runtime_error("un-recognized cmdline arg '" + arg + "'");
        }
        if (inFileName == "")
          throw std::runtime_error("no input file specified...");
        if (outFileName == "")
          throw std::runtime_error("no output file specified...");
        if (largeBricks && spatialMedianBuilder)
          throw std::runtime_error("you gotta decide, either spatial median _or_ large bricks...");
        SingleCellModel::SP input = loadExaJet(inFileName);
        int builderType = !spatialMedianBuilder && !largeBricks?
                SAH_ALIKE_BUILDER : largeBricks?
                SMALL_BRICK_COUNT_BUILDER: SPATIAL_MEDIAN_BUILDER;
        Bricker bricker(input,builderType);
        BrickedModel::SP output = bricker.output;
        std::cout << "Done bricking, created " << output->bricks.size() << " bricks" << std::endl;

        // Compute some stats
        if (1) {
          vec3d avgCellCount(0.);
          int numSingleCellBricks = 0;
          for (auto brick : output->bricks) {
            avgCellCount += vec3d(brick->cellIDs.size);
            if (brick->cellIDs.size == vec3i(1)) {
              ++numSingleCellBricks;
            }
          }
          avgCellCount /= vec3d(output->bricks.size());
          std::cout << "Average num cells per brick : " << avgCellCount << '\n';
          std::cout << "Number of single-cell bricks: " << numSingleCellBricks << '\n';
        }

        std::ofstream out(outFileName);
        for (auto brick : output->bricks) {
          out.write((char*)& brick->cellIDs.size, sizeof(brick->cellIDs.size));
          out.write((char*)& brick->lower, sizeof(brick->lower));
          out.write((char*)& brick->level, sizeof(brick->level));
          out.write((char*)brick->cellIDs.elements.data(),
                    sizeof(id_t) * owl::volume(brick->cellIDs.size));
        }

        if (!kdFileName.empty()) {
          std::ofstream kdout(kdFileName);
          kdout.write((char*)bricker.tree.data(), sizeof(exa::KdTreeNode) * bricker.tree.size());
        }

        return 0;
      }
      catch (std::runtime_error e) {
        std::cerr << "FATAL Error : " << e.what() << std::endl;
        exit(1);
      }
    }
  
  };
