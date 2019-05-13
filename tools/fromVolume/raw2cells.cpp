#include "owl/common/parallel/parallel_for.h"
#include "owl/common/math/box.h"
#include <vector>
#include <fstream>
// stdlib, for mmap
#include <sys/types.h>
#include <sys/stat.h>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif
#include <fcntl.h>
#include <string>
#include <cstring>

// O_LARGEFILE is a GNU extension.
#ifdef __APPLE__
#define  O_LARGEFILE  0
#endif


#ifdef _WIN32
# define MMAP 0
#else
# define MMAP 1
#endif

using namespace owl;

struct Cell {
  vec3i lower;
  int   level;
  float scalar;
};

std::ofstream cellFile;
std::ofstream scalarFile;

// std::vector<Cell> result;
std::mutex resultMutex;

int maxLevel = 6;
int refinement = 2;

template<typename T>
struct Volume {
  Volume(const std::string &inFileName,
         const vec3i &dims)
    : dims(dims)
  {
#if MMAP
    // mmap the binary file
    FILE *file = fopen(inFileName.c_str(),"rb");
    if (!file)
      throw std::runtime_error("could not mmap input file "+inFileName);
    fseek(file,0,SEEK_END);
    size_t actualFileSize = ftell(file);
    fclose(file);
      
    int fd = ::open(inFileName.c_str(), O_LARGEFILE | O_RDONLY);
    assert(fd >= 0);
    
    unsigned char *mem = (unsigned char *)mmap(NULL,actualFileSize,PROT_READ,MAP_SHARED,fd,0);
    assert(mem != NULL && (long long)mem != -1LL);
    elt = (T*)mem;
#else
    std::ifstream in(inFileName);
    for (int iz=0;iz<dims.z;iz++)
      for (int iy=0;iy<dims.y;iy++)
        for (int ix=0;ix<dims.x;ix++) {
          T t;
          in.read((char*)&t,sizeof(t));
          elt.push_back(t);
        }
#endif
  }

  inline T operator[](const vec3i &idx) const
  {
    return elt[idx.x+dims.x*(idx.y+dims.y*((size_t)idx.z))];
  }
  
  interval<T> range(const box3i &coords) const
  {
    interval<T> result;
    for (int iz=coords.lower.z;iz<std::min(dims.z,coords.upper.z);iz++) 
      for (int iy=coords.lower.y;iy<std::min(dims.y,coords.upper.y);iy++) 
        for (int ix=coords.lower.x;ix<std::min(dims.x,coords.upper.x);ix++) {
          result.extend((*this)[vec3i(ix,iy,iz)]);
        }
    return result;
  }

  T average(const box3i &coords) const
  {
    T sum = 0.f;
    for (int iz=coords.lower.z;iz<std::min(dims.z,coords.upper.z);iz++) 
      for (int iy=coords.lower.y;iy<std::min(dims.y,coords.upper.y);iy++) 
        for (int ix=coords.lower.x;ix<std::min(dims.x,coords.upper.x);ix++) {
          sum += ((*this)[vec3i(ix,iy,iz)]);
        }
    return sum / (float)volume(coords);
  }
  
  const vec3i dims;
#if MMAP
  T *elt;
#else
  std::vector<T> elt;
#endif
};


/*! number of logical-grid cell written out, across all levels */
size_t numLogicalWritten = 0;

void write(const Cell &cell)
{
  static size_t nextPing = 1;
  static size_t numWritten = 0;
  static box4i bounds;

  numWritten++;
  vec4i ci(cell.lower,cell.level);
  bounds.extend(ci);
  cellFile.write((char *)&ci,sizeof(ci));
  scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));
  numLogicalWritten += (1ull << (3*cell.level));
  if (numWritten >= nextPing) {
    std::cout << "just written " << ci << std::endl;
    std::cout << "num written : " << prettyNumber(numWritten) << ", bounds " << bounds << ", voxels " << prettyNumber(numLogicalWritten) << std::endl;
    nextPing *= 2;
  }
}
    // cellFile.write((char *)&cell.lower,sizeof(cell.lower));
    // cellFile.write((char *)&cell.level,sizeof(cell.level));
    // scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));

template<typename T>
void buildCell(const Volume<T> &volume,
               const vec3i &lower,
               int level,
               const float threshold)
{
  if (lower.x >= volume.dims.x) return;
  if (lower.y >= volume.dims.y) return;
  if (lower.z >= volume.dims.z) return;

  if (level == 0) {
    Cell cell;
    cell.lower = lower;
    cell.level = level;
    cell.scalar = (float)volume[lower];
    std::lock_guard<std::mutex> lock(resultMutex);
    // result.push_back(cell);
    write(cell);
    // cellFile.write((char *)&cell.lower,sizeof(cell.lower));
    // cellFile.write((char *)&cell.level,sizeof(cell.level));
    // scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));
    // cellFile.write((char *)&cell.lower,sizeof(cell.lower));
    // cellFile.write((char *)&cell.level,sizeof(cell.level));
    // scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));
    
    return;
  }

  int childCellWidth = 1;
  for (int i=0;i<(level-1);i++)
    childCellWidth *= refinement;
  int cellWidth = childCellWidth * refinement;

  box3i cellBounds(lower,lower+cellWidth);
  owl::interval<T> range = volume.range(cellBounds);
  if (level == maxLevel)
    std::cout << "building root-brick at " << cellBounds << " range " << (float)range.lower << ".." << (float)range.upper << std::endl;
  // if ((level+1) == maxLevel)
  //   std::cout << " - level-"<<(level)<<" sub-brick at " << cellBounds << " range " << (float)range.lower << ".." << (float)range.upper << std::endl;
  // if ((level+2) == maxLevel) {
  //   std::cout << "\n   - level-"<<(level)<<" sub-brick at " << cellBounds << " range " << (float)range.lower << ".." << (float)range.upper << std::endl;
  //   // PRINT(range);
  //   // PRINT(threshold);
  // }

  
  if ((range.upper-range.lower) <= threshold) {
    // std::cout << level << std::flush;
    // std::cout << "level-" << level << " leaf" << std::endl;
    Cell cell;
    cell.lower = lower;
    cell.level = int(log2f(cellWidth)); // WRONG: level;
    cell.scalar = volume.average(cellBounds);//0.5f*(range.upper-range.lower);//(float)volume[lower];
    std::lock_guard<std::mutex> lock(resultMutex);
    // result.push_back(cell);
    write(cell);
    // cellFile.write((char *)&cell.lower,sizeof(cell.lower));
    // cellFile.write((char *)&cell.level,sizeof(cell.level));
    // scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));
    return;
  }

  // int hcw = (1<<(level-1));
  for (int iz=0;iz<refinement;iz++)
    for (int iy=0;iy<refinement;iy++)
      for (int ix=0;ix<refinement;ix++) {
        buildCell(volume,lower+vec3i(ix,iy,iz)*childCellWidth,level-1,threshold);
      }
}

template<typename T>
void makeCells(const vec3i &dims,
               const std::string &inFileName,
               float threshold)
{
  Volume<T> input(inFileName,dims);
  int rootCellWidth = 1;
  for (int i=0;i<maxLevel;i++)
    rootCellWidth *= refinement;
  PRINT(rootCellWidth);
  vec3i numRootCells = divRoundUp(dims,vec3i(rootCellWidth));
  owl::serial_for(volume(numRootCells),[&](int64_t cellID){
      vec3i idx = vec3i(int(cellID % numRootCells.x),
                        int((cellID/numRootCells.x) % numRootCells.y),
                        int(cellID/numRootCells.x/numRootCells.y));
      buildCell(input,idx*rootCellWidth,maxLevel,threshold);

      static size_t numOutputCellsCreated = 0;
      static size_t numInputCellsProcessed = 0;

      std::lock_guard<std::mutex> lock(resultMutex);
      box3i brickRegion(idx*rootCellWidth,
                        min((idx+vec3i(1))*rootCellWidth,dims));
      // PRINT(brickRegion);
      numOutputCellsCreated
        = (scalarFile.tellp() / sizeof(float));
      numInputCellsProcessed
        += brickRegion.volume();
      std::cout << "done brick " << brickRegion
                << " : " << " now total of " << prettyNumber(numOutputCellsCreated)
                << " output cells for " << prettyNumber(numInputCellsProcessed)
                << " input cells, that is a compression of "
                << int(100.f-numOutputCellsCreated*100.f/numInputCellsProcessed)
                << "%" << std::endl;
    });
}

int main(int ac, char **av)
{
  if (ac != 9 && ac != 10)
    throw std::runtime_error("usage: exaRawToCells nx ny nz float inFile.raw outFile.exa threshold maxLevel [refinement]");
  vec3i dims = vec3i(atoi(av[1]),atoi(av[2]),atoi(av[3]));
  const std::string format = av[4];
  const std::string inFileName = av[5];
  const std::string outFileName = av[6];
  const float threshold = std::stof(av[7]);
  maxLevel = atoi(av[8]);
  if (ac == 10)
    refinement = atoi(av[9]);
  
  cellFile = std::ofstream(outFileName+".cells");
  scalarFile = std::ofstream(outFileName+".scalars");
  if (format == "byte")
    makeCells<uint8_t>(dims,inFileName,threshold);
  else if (format == "float")
    makeCells<float>(dims,inFileName,threshold);
  else if (format == "double")
    makeCells<double>(dims,inFileName,threshold);
  else throw std::runtime_error("unknown format!?");
  size_t numCells = scalarFile.tellp() / sizeof(float);
  std::cout << "done creating cells, found " << prettyNumber(numCells) << " cells" << std::endl;

  std::cout << "that is " << int(numCells*100.f/volume(dims)) << "% of the input" << std::endl;
  
  std::cout << "total num logical written " << prettyNumber(numLogicalWritten) << ", expected " << prettyNumber(dims.x*size_t(dims.y)*dims.z) << std::endl;
  // std::ofstream cellFile(outFileName+".cells");
  // std::ofstream scalarFile(outFileName+".scalars");
  // for (auto cell : result) {
  //   scalarFile.write((char *)&cell.scalar,sizeof(cell.scalar));
    
  //   cellFile.write((char *)&cell.lower,sizeof(cell.lower));
  //   cellFile.write((char *)&cell.level,sizeof(cell.level));
  // }
}
