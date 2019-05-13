#include <vtkGenericDataObjectReader.h>
#include <vtkStructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <string>
#include "gdt/math/vec.h"
#include "gdt/math/box.h"
#include <vector>
#include <map>

using namespace gdt;

// std::vector<std::vector<vec3f>> vertexVecVec;
// std::vector<std::vector<vec3i>> indexVecVec;

int main ( int argc, char *argv[] )
{
  // Ensure a filename was specified
  // if(argc != 2)
  // {
  //   std::cerr << "Usage: " << argv[0] << " InputFilename" << endl;
  //   return EXIT_FAILURE;
  // }

  std::ofstream out("out.tris");
  for (int fileID=1;fileID<argc;fileID++) {
    // Get the filename from the command line
    std::string inputFilename = argv[fileID];

    std::vector<vec3f> vertexVec;
    std::vector<vec3i> indexVec;
 
    // Get all data from the file
    vtkSmartPointer<vtkGenericDataObjectReader> reader =
      vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFilename.c_str());
    reader->Update();

    // All of the standard data types can be checked and obtained like this:
    if(!reader->IsFilePolyData())
      continue;
    
    // std::cout << "output is a polydata" << std::endl;
    vtkPolyData* output = reader->GetPolyDataOutput();
    int numPoints = output->GetNumberOfPoints();
    std::cout << "file has " << output->GetNumberOfPoints() << " points." << std::endl;
    for (int pointID=0;pointID<numPoints;pointID++) {
      double point[3];
      output->GetPoint(pointID,point);

      vertexVec.push_back(vec3f(point[0],point[1],point[2]));
      // for (int i=0;i<3;i++) {
      //   printf("%f %f %f\n",(float)point[0],(float)point[1],(float)point[2]);
      // }
    }


    vtkCellArray *cells = output->GetPolys();
    int polys = output->GetNumberOfCells();//cells->GetSize();
    printf("file has %i polys\n",polys);
    for (int i=0;i<polys;i++) {
      vtkCell *cell = output->GetCell(i);
      if (!cell) continue;
      // vtkIdType *idx = nullptr; //[10];
      // vtkIdType npts = 10;
      // output->GetCell((vtkIdType)i,npts,idx);
      // printf("tri %i : %i %i %i\n",npts,idx[0],idx[1],idx[2]);
      // vtkIndent indent;
      // cell->PrintSelf(std::cout,indent);

      int numVertices = cell->GetNumberOfPoints();
      // printf("vertices %i\n",numVertices);
      if (numVertices == 0) continue;
      if (numVertices == 3) {
        int x = cell->GetPointId(0);
        int y = cell->GetPointId(1);
        int z = cell->GetPointId(2);
        indexVec.push_back(vec3i(x,y,z));
        // printf("triangle %i %i %i\n",x,y,z);
        continue;
      }
      if (numVertices == 4) {
        int x = cell->GetPointId(0);
        int y = cell->GetPointId(1);
        int z = cell->GetPointId(2);
        int w = cell->GetPointId(3);
        indexVec.push_back(vec3i(x,y,z));
        indexVec.push_back(vec3i(x,z,w));
        // printf("quad %i %i %i %i\n",x,y,z,w);
        continue;
      }
      printf("unsupported num vertices %i\n",numVertices);
      exit(1);
    }
    
    std::vector<vec3f> remappedVertex;
    std::map<vec3f,int> vertexMapper;
    for (int i=0;i<vertexVec.size();i++) {
      const vec3f org_v = vertexVec[i];
      if (vertexMapper.find(org_v) != vertexMapper.end())
        continue;
      vertexMapper[org_v] = remappedVertex.size();
      remappedVertex.push_back(org_v);
    }
    
    for (int i=0;i<indexVec.size();i++) {
      indexVec[i].x = vertexMapper[vertexVec[indexVec[i].x]];
      indexVec[i].y = vertexMapper[vertexVec[indexVec[i].y]];
      indexVec[i].z = vertexMapper[vertexVec[indexVec[i].z]];
    }

#if 0

    DO NOT USE THIS - TRANSFORMING *THIS* WAY WILL LEAD TO NUMERICALLY
      TOTALLY UNSTABLE VERTEX POSITIONS. WE *HAVE* TO WORK IN WORLD SPACE AND TRANSFORM RAYS
      TO GRID SPACE FOR SAMPLING, NOT THE OHTER WAY AROUND

    
    // exajet: remap from world space to grid space ... should actually
    // map rays to grid space, and leave triangles in world
    // space.... but for now, that'll have to do
    for (auto &vtx : remappedVertex) {
      // # bbi: ((1232128, 1259072, 1238336), (1270848, 1277952, 1255296))
      // # bbr: ((-1.73575, -9.44, -3.73281), (17.6243, 0, 4.74719))
    
      const box3f bbi(vec3f(1232128, 1259072, 1238336),
                      vec3f(1270848, 1277952, 1255296));
      const box3f bbr(vec3f(-1.73575, -9.44, -3.73281),
                      vec3f(17.6243, 0, 4.74719));
    
      // normalize to [0,1]^3
      vtx = (vtx - bbr.lower) / (bbr.upper - bbr.lower);
      // and cale back to grid space
      vtx = bbi.lower + vtx * (bbi.upper - bbi.lower);
    }
#endif
  //   vertexVecVec.push_back(vertexVec);
  //   indexVecVec.push_back(indexVec);
  // }
  
    int numVertices = remappedVertex.size();
    int numTriangles = indexVec.size();
    
    out.write((char*)&numVertices,sizeof(numVertices));
    out.write((char *)remappedVertex.data(),remappedVertex.size()*sizeof(vec3f));
    PRINT(out.tellp());
    out.write((char*)&numTriangles,sizeof(numTriangles));
    out.write((char *)indexVec.data(),indexVec.size()*sizeof(vec3i));
  }
  std::cout << "written all meshes ..." << std::endl;
  return EXIT_SUCCESS;
}
