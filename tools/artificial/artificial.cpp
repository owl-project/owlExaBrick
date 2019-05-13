#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "../fromFlash/Cmdline.h"

// v0 --- x ----- v1
inline float lerp(float v0, float v1, float x)
{
    return (1.f - x) * v0 + x * v1;
}

inline float bilerp(float v0, float v1, float v2, float v3, float x, float y)
{
    return lerp(lerp(v0,v1,x),lerp(v2,v3,x),y);
}

inline float trilerp(float v0, float v1, float v2, float v3,
                     float v4, float v5, float v6, float v7,
                     float x, float y, float z)
{
    return lerp(bilerp(v0,v1,v2,v3,x,y), bilerp(v4,v5,v6,v7,x,y), z);
}

inline float trilerp(float const* vals, float x, float y, float z)
{
    return trilerp(vals[0],vals[1],vals[2],vals[3],
                   vals[4],vals[5],vals[6],vals[7],
                   x,y,z);
}


struct Grid
{
    int minCorner[3];
    int nx, ny, nz;
    int level;
    float scalarPerVertex[8];


    Grid() = default;
    Grid(std::array<int,3> minc, int nx, int ny, int nz, int l, std::array<float,8> scalars)
        : nx(nx)
        , ny(ny)
        , nz(nz)
        , level(l)
    {
        memcpy(&minCorner,minc.data(),sizeof(minCorner));
        memcpy(&scalarPerVertex,scalars.data(),sizeof(scalarPerVertex));
    }

    void appendFlattened(std::vector<int>& cells, std::vector<float>& scalars)
    {
        int cellsize = 1<<level;

        int maxCorner[3] = {
            minCorner[0]+(nx-1)*cellsize,
            minCorner[1]+(ny-1)*cellsize,
            minCorner[2]+(nz-1)*cellsize,
            };

        for (int cz=minCorner[2]; cz<=maxCorner[2]; cz += cellsize)
        {
            for (int cy=minCorner[1]; cy<=maxCorner[1]; cy += cellsize)
            {
                for (int cx=minCorner[0]; cx<=maxCorner[0]; cx += cellsize)
                {
                    float x = (cx-minCorner[0])/((float)maxCorner[0]-minCorner[0]+1);
                    float y = (cy-minCorner[1])/((float)maxCorner[1]-minCorner[1]+1);
                    float z = (cz-minCorner[2])/((float)maxCorner[2]-minCorner[2]+1);

                    cells.push_back(cx);
                    cells.push_back(cy);
                    cells.push_back(cz);
                    cells.push_back(level);
                    scalars.push_back(trilerp(scalarPerVertex,x,y,z));

                    // std::cout << cx << ',' << cy << ',' << cz << ',' << level << ',' << scalars.back() << '\n';
                }
            }
        }
    }
};


int main(int argc, char** argv)
{
    std::string fileName = "";
    std::string outName = "artificial";

    cl::Cmdline cli("exaArtificial", "Create artificial exa data sets");

    cli.Add(
        "fileName",
        "fileName (contains list of subgrids. One subgrid per line. Either \n\"min0,min1,min2,nx,ny,nz,level,value\" or\n"
        "\"min0,min1,min2,nx,ny,nz,level,v0,v1,v2,v3,v4,v5,v6,v7)\"",
        cl::Required::yes | cl::Arg::yes | cl::Positional::yes,
        cl::Var(fileName)
        );

    cli.Add(
        "o",
        "Output base filename (default: artificial)",
        cl::Required::no | cl::Arg::required,
        cl::Var(outName)
        );

    auto result = cli.Parse(argv+1,argv+argc);

    cli.PrintDiag();

    if (!result)
    {
        cli.PrintHelp();
        return EXIT_FAILURE;
    }

    std::fstream gridFile(fileName);
    std::vector<int> cells;
    std::vector<float> scalars;

    std::string line;
    while (std::getline(gridFile, line))
    {
        Grid grid;
        bool valid = false;
        if (sscanf(line.c_str(),"%i %i %i %i %i %i %i %f %f %f %f %f %f %f %f",
                   &grid.minCorner[0],&grid.minCorner[1],&grid.minCorner[2],
                   &grid.nx,&grid.ny,&grid.nz,&grid.level,
                   &grid.scalarPerVertex[0],
                   &grid.scalarPerVertex[1],
                   &grid.scalarPerVertex[2],
                   &grid.scalarPerVertex[3],
                   &grid.scalarPerVertex[4],
                   &grid.scalarPerVertex[5],
                   &grid.scalarPerVertex[6],
                   &grid.scalarPerVertex[7]) == 15)
        {
            valid = true;
        }
        else if (sscanf(line.c_str(),"%i %i %i %i %i %i %i %f",
                        &grid.minCorner[0],&grid.minCorner[1],&grid.minCorner[2],
                        &grid.nx,&grid.ny,&grid.nz,&grid.level,
                        &grid.scalarPerVertex[0]) == 8)
        {
            for (int i=1; i<8; ++i)
            {
                grid.scalarPerVertex[i] = grid.scalarPerVertex[0];
            }

            valid = true;
        }

        if (valid)
            grid.appendFlattened(cells,scalars);
    }

    if (cells.empty() || scalars.empty())
        return EXIT_FAILURE;

    std::string cellFileName = outName+".cells";
    std::string scalarFileName = outName+".scalars";

    std::fstream cellFile(cellFileName,std::fstream::out|std::fstream::binary);
    std::fstream scalarFile(scalarFileName,std::fstream::out|std::fstream::binary);

    if (!cellFile.good() || !scalarFile.good())
        return EXIT_FAILURE;

    std::cout << "Writing data to " << cellFileName << " and " << scalarFileName << '\n';
    cellFile.write((char*)cells.data(), cells.size()*sizeof(int));
    scalarFile.write((char*)scalars.data(), scalars.size()*sizeof(float));

    return EXIT_SUCCESS;
}
