
/* iw - quick tool to convert from vtk vtu format to ugrid64 format */

#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>

#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkNamedColors.h>

#include "owl/common/math/box.h"
#include <set>

using namespace owl;

std::vector<double> vertex;
std::vector<std::string> scalarArrayName;
std::vector<std::vector<double>> scalarArrayData; // one vector per field
std::vector<size_t> hex_index;

#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << (var) << std::endl;
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif

void readFile(const std::string fileName)
{
  std::cout << "parsing vtu file " << fileName << std::endl;
  //read all the data from the file
  vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
    vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName(fileName.c_str());
  reader->Update();

  vtkUnstructuredGrid *grid = reader->GetOutput();
  
  vtkPointData* pointData = grid->GetPointData();
  // pointData->PrintSelf(std::cout,vtkIndent());
  size_t firstIndexThisVTU = vertex.size() / 3;
  
  // ==================================================================
  const int numPoints = grid->GetNumberOfPoints();
  std::cout << " - found " << numPoints << " points" << std::endl;
  for (int pointID=0;pointID<numPoints;pointID++) {
    double point[3];
    grid->GetPoint(pointID,point);
    for (int i=0;i<3;i++)
      vertex.push_back(point[i]);
  }

  // ==================================================================
  const int numCells = grid->GetNumberOfCells();
  std::cout << " - found " << numCells << " cells" << std::endl;
  for (int cellID=0;cellID<numCells;cellID++) {
    vtkIdType cellPoints;
    vtkIdType *pointIDs;
    grid->GetCellPoints(cellID,cellPoints,pointIDs);

    if (cellPoints == 8) {
      for (int i=0;i<cellPoints;i++) {
        int vtxID = pointIDs[i];
        if (vtxID < 0 || vtxID >= numPoints) {
          PING;
          PRINT(vtxID);
          for (int j=0;j<8;j++)
            PRINT(pointIDs[j]);
          exit(0);
        }
        hex_index.push_back(firstIndexThisVTU + pointIDs[i]);
      }
    } else
      throw std::runtime_error("unsupported number of points per cell : "
                               +std::to_string((int)numPoints));
    
    // std::cout << " cell N=" << numPoints << " { ";
    // for (int i = 0; i<numPoints; i++)
    //   std::cout << pointIDs[i] << " ";
    // std::cout << "}" << std::endl;
  }

  vtkCellData* cellData = grid->GetCellData();
  if (!cellData)
    throw std::runtime_error("could not read cell data ....");
  
  // cellData->PrintSelf(std::cout,vtkIndent());
  // std::cout << "==================================================================" << std::endl;
  if (scalarArrayData.empty()) {
    scalarArrayData.resize(cellData->GetNumberOfArrays());
    scalarArrayName.resize(cellData->GetNumberOfArrays());
  }
  for (int i = 0; i < cellData->GetNumberOfArrays(); i++) {
    scalarArrayName[i] = cellData->GetArrayName(i);
    std::cout << "\tArray " << i
              << " is named "
              << (cellData->GetArrayName(i) ? cellData->GetArrayName(i) : "NULL")
              << std::endl;

    vtkDataArray *dataArray = cellData->GetArray(i);
    if (!dataArray)
      throw std::runtime_error("could not read data array from cell data");
    for (int j=0;j<numCells;j++)
      scalarArrayData[i].push_back(dataArray->GetTuple1(j));
    // std::cout << "   dat[" << i << "] = " << dataArray->GetTuple1(i) << std::endl;
    // std::cout << "==================================================================" << std::endl;
    // }
    
  }
  std::cout << "-------------------------------------------------------" << std::endl;
  std::cout << "done reading " << fileName << " : "
            << std::endl << "  " << (vertex.size()/3) << " vertices "
            << std::endl << "  " << (hex_index.size()/8) << " hexes" << std::endl;
  
}


int project(double f)
{
  // apparently in the deep impact model all vertices are multiples of
  // 5*500....
  const int commonFactor = (5*500) / 4;
  if (fmodf(f,commonFactor) != 0) {
    PRINT(f);
    PRINT(commonFactor);
    throw std::runtime_error("not a multiple of commonFactor");
  }
  int asInt = int(f/commonFactor);
  if (asInt*commonFactor != f)
    throw std::runtime_error("error2: not a multiple of commonFactor");
  return asInt;
}

vec3i project(const vec3d v)
{
  return vec3i(project(v.x),
               project(v.y),
               project(v.z));
}

struct Hex {
  size_t idx[8];
};

struct SingleCell {
  vec3i lower;
  int   level;
};

int main ( int argc, char *argv[] )
{
  std::string outFileName = "";
  std::vector<std::string> inFileNames;
  size_t maxFiles = 1ULL<<60;
  
  for (int i=1;i<argc;i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      outFileName = argv[++i];
    else if (arg == "--max-files")
      maxFiles = atol(argv[++i]);
    else
      inFileNames.push_back(arg);
  }
  
  if(inFileNames.empty() || outFileName.empty())
  {
    std::cerr << "Usage: " << argv[0] << " -o outfile <infiles.vtu>+" << std::endl;
    return EXIT_FAILURE;
  }

  if (inFileNames.size() > maxFiles)
    inFileNames.resize(maxFiles);

  for (int fileID=0;fileID<inFileNames.size();fileID++) {
  // for (auto fn : inFileNames)
    const std::string fn = inFileNames[fileID];
    readFile(fn);
    std::cout << "done reading file no " << fileID << "/" << inFileNames.size() << " (" << int((fileID+1)*100.f/inFileNames.size()) << "%)" << std::endl;
  }

  // std::cout << "=======================================================" << std::endl;
  // std::cout << "computing per-vertex data from per cell data ..." << std::endl;
  // std::cout << "=======================================================" << std::endl;

  int numVertices = vertex.size() / 3;
  // std::vector<float> perVertexValue(numVertices);
  // std::vector<float> perVertexCount(numVertices);
  // for (int i=0;i<hex_index.size();i++) {
  //   int cellID = i/8;
  //   int vtxID = hex_index[i];
  //   if (vtxID == -1) continue;
  //   if (vtxID < 0 || vtxID >= perVertexValue.size()) {
  //     PING;
  //     PRINT(vtxID);
  //     PRINT(perVertexValue.size());
  //   }
  //   float cellValue = perCellValue[cellID];
  //   perVertexValue[vtxID] += cellValue;
  //   perVertexCount[vtxID] += 1.f;
  // }
  // for (int i=0;i<numVertices;i++)
  //   perVertexValue[i] /= (perVertexCount[i] + 1e-20f);

  // struct {
  //   size_t n_verts, n_tris, n_quads, n_tets, n_pyrs, n_prisms, n_hexes;
  // } header;
  // header.n_verts  = numVertices;
  // header.n_tris   = 0;
  // header.n_quads  = 0;
  // header.n_tets   = 0;
  // header.n_pyrs   = 0;
  // header.n_prisms = 0;
  // header.n_hexes  = hex_index.size()/8;

  // std::cout << "=======================================================" << std::endl;
  // std::cout << "writing out result ..." << std::endl;
  // std::cout << "=======================================================" << std::endl;
  // std::ofstream out(outFileName+".ugrid64");
  // out.write((char*)&header,sizeof(header));
  // out.write((char*)vertex.data(),vertex.size()*sizeof(vertex[0]));
  // for (int i=0;i<vertex.size();i++)
  //   PRINT(vertex[i]);

  size_t numHexes = hex_index.size()/8;
  vec3d *vertexArray = (vec3d*)vertex.data();
  Hex *hexArray = (Hex *)hex_index.data();

  PRINT(numHexes);
  PRINT(numVertices);

  std::ofstream cellStream(outFileName+".cells");

  for (int i=0;i<numHexes;i++) {
    std::set<int> set_x, set_y, set_z;
    Hex hex = hexArray[i];
    box3i bounds;
    for (int j=0;j<8;j++) {
      int vtxID = hex.idx[j];
      vec3i vtx = project(vertexArray[vtxID]);
      bounds.extend(vtx);
      set_x.insert(vtx.x);
      set_y.insert(vtx.y);
      set_z.insert(vtx.z);
    }
    // PING;
    // PRINT(bounds);
    // PRINT(set_x.size());
    // PRINT(set_y.size());
    // PRINT(set_z.size());
    if (set_x.size() != 2 ||
        set_y.size() != 2 ||
        set_z.size() != 2) {
      PING;
      PRINT(bounds);
      PRINT(bounds.size());
      int szx = bounds.size().x;
      for (int j=0;j<8;j++) {
        int vtxID = hex.idx[j];
        vec3i vtx = project(vertexArray[vtxID]);
        PRINT(vtx);
      }
      for (int j=0;j<8;j++) {
        int vtxID = hex.idx[j];
        vec3i vtx = project(vertexArray[vtxID]);
        PRINT((vtx.x & (szx-1)));
        PRINT((vtx.y & (szx-1)));
        PRINT((vtx.z & (szx-1)));
      }
      throw std::runtime_error("vertices do not form a regular hex!");
    }
    vec3i size = bounds.size();
    if (!((size.x == size.y) && (size.x == size.z)))
      throw std::runtime_error("not a cubic cell...");
    
    int cellWidth = size.x;
    int level = int(log2(cellWidth));
    if ((1<<level) != cellWidth)
      throw std::runtime_error("error in computing cell width");

    SingleCell cell;
    cell.level = level;
    cell.lower = bounds.lower;

    cellStream.write((char*)&cell,sizeof(cell));
    // std::cout << "cell " << bounds.lower << " size " << bounds.size() << std::endl;
  }
  std::cout << "Done. Written " << numHexes << " cells..." << std::endl;

  for (int j=0;j<scalarArrayName.size();j++) {
    std::cout << "writing scalar array " << scalarArrayName[j] << std::endl;
    std::ofstream scalarStream(outFileName+"."+scalarArrayName[j]+".scalars");
    for (int i=0;i<numHexes;i++) {
      float scalar = scalarArrayData[j][i];
      scalarStream.write((char*)&scalar,sizeof(scalar));
    }
  }

  std::ofstream obj("test.obj");
  for (int i=0;i<numVertices;i++)
    obj << "v " << vertexArray[i].x << " " << vertexArray[i].y << " " << vertexArray[i].z << std::endl;
  for (int i=0;i<numHexes;i++) {
    Hex hex = hexArray[i];

    int vtx[8];
    for (int j=0;j<8;j++)
      vtx[j] = hex.idx[j]+i;
    
    obj << "f " << vtx[0] << " " << vtx[1] << " " << vtx[2] << " " << vtx[3] << std::endl;
  }
  
  // for (auto &idx : hex_index) idx += 1;
  // out.write((char*)hex_index.data(),hex_index.size()*sizeof(hex_index[0]));
  
  // std::ofstream value_out(outFileName+".scalar");
  // value_out.write((char*)perVertexValue.data(),perVertexValue.size()*sizeof(perVertexValue[0]));
  
  return EXIT_SUCCESS;
}
