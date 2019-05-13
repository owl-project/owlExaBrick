#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <H5Cpp.h>

#include "Cmdline.h"

#define MAX_STRING_LENGTH 80

struct sim_info_t
{
    int file_format_version;
    char setup_call[400];
    char file_creation_time[MAX_STRING_LENGTH];
    char flash_version[MAX_STRING_LENGTH];
    char build_date[MAX_STRING_LENGTH];
    char build_dir[MAX_STRING_LENGTH];
    char build_machine[MAX_STRING_LENGTH];
    char cflags[400];
    char fflags[400];
    char setup_time_stamp[MAX_STRING_LENGTH];
    char build_time_stamp[MAX_STRING_LENGTH];
};

struct grid_t
{
    typedef std::array<char, 4> char4;
    typedef struct __attribute__((packed)) { double x, y, z; } vec3d;
    typedef struct { vec3d min, max; } aabbd;

    struct __attribute__((packed)) gid_t
    {
        int neighbors[6];
        int parent;
        int children[8];
    };

    std::vector<char4> unknown_names;
    std::vector<int> refine_level;
    std::vector<int> node_type; // node_type 1 ==> leaf
    std::vector<gid_t> gid;
    std::vector<vec3d> coordinates;
    std::vector<vec3d> block_size;
    std::vector<aabbd> bnd_box;
    std::vector<int> which_child;
};

struct variable_t
{
    size_t global_num_blocks;
    size_t nxb;
    size_t nyb;
    size_t nzb;

    std::vector<double> data;
};

void read_sim_info(sim_info_t& dest, H5::H5File const& file)
{
    H5::StrType str80(H5::PredType::C_S1, 80);
    H5::StrType str400(H5::PredType::C_S1, 400);

    H5::CompType ct(sizeof(sim_info_t));
    ct.insertMember("file_format_version", 0, H5::PredType::NATIVE_INT);
    ct.insertMember("setup_call", 4, str400);
    ct.insertMember("file_creation_time", 404, str80);
    ct.insertMember("flash_version", 484, str80);
    ct.insertMember("build_date", 564, str80);
    ct.insertMember("build_dir", 644, str80);
    ct.insertMember("build_machine", 724, str80);
    ct.insertMember("cflags", 804, str400);
    ct.insertMember("fflags", 1204, str400);
    ct.insertMember("setup_time_stamp", 1604, str80);
    ct.insertMember("build_time_stamp", 1684, str80);

    H5::DataSet dataset = file.openDataSet("sim info");

    dataset.read(&dest, ct);
}

void read_grid(grid_t& dest, H5::H5File const& file)
{
    H5::DataSet dataset;
    H5::DataSpace dataspace;

    {
        H5::StrType str4(H5::PredType::C_S1, 4);

        dataset = file.openDataSet("unknown names");
        dataspace = dataset.getSpace();
        dest.unknown_names.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.unknown_names.data(), str4, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("refine level");
        dataspace = dataset.getSpace();
        dest.refine_level.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.refine_level.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("node type");
        dataspace = dataset.getSpace();
        dest.node_type.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.node_type.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("gid");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.gid.resize(dims[0]);
        assert(dims[1] == 15);

        dataset.read(dest.gid.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("coordinates");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.coordinates.resize(dims[0]);
        assert(dims[1] == 3);

        dataset.read(dest.coordinates.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("block size");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.block_size.resize(dims[0]);
        assert(dims[1] == 3);

        dataset.read(dest.block_size.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("bounding box");
        dataspace = dataset.getSpace();

        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims);
        dest.bnd_box.resize(dims[0] * 2);
        assert(dims[1] == 3);
        assert(dims[2] == 2);

        std::vector<double> temp(dims[0] * dims[1] * dims[2]);

        dataset.read(temp.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);

        dest.bnd_box.resize(dims[0]);
        for (size_t i = 0; i < dims[0]; ++i)
        {
            dest.bnd_box[i].min.x = temp[i * 6];
            dest.bnd_box[i].max.x = temp[i * 6 + 1];
            dest.bnd_box[i].min.y = temp[i * 6 + 2];
            dest.bnd_box[i].max.y = temp[i * 6 + 3];
            dest.bnd_box[i].min.z = temp[i * 6 + 4];
            dest.bnd_box[i].max.z = temp[i * 6 + 5];
        //std::cout << dest.bnd_box[i].min.x << ' ' << dest.bnd_box[i].min.y << ' ' << dest.bnd_box[i].min.z << '\n';
        //std::cout << dest.bnd_box[i].max.x << ' ' << dest.bnd_box[i].max.y << ' ' << dest.bnd_box[i].max.z << '\n';
        }
    }

    {
        dataset = file.openDataSet("which child");
        dataspace = dataset.getSpace();
        dest.which_child.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.which_child.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }
}

void read_variable(variable_t& var, H5::H5File const& file, char const* varname)
{
    H5::DataSet dataset = file.openDataSet(varname);
    H5::DataSpace dataspace = dataset.getSpace();

    //std::cout << dataspace.getSimpleExtentNdims() << '\n';
    hsize_t dims[4];
    dataspace.getSimpleExtentDims(dims);
    var.global_num_blocks = dims[0];
    var.nxb = dims[1];
    var.nyb = dims[2];
    var.nzb = dims[3];
    var.data.resize(dims[0] * dims[1] * dims[2] * dims[3]);
    dataset.read(var.data.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    //std::cout << dims[0] << ' ' << dims[1] << ' ' << dims[2] << ' ' << dims[3] << '\n';
}

void make_cells(grid_t const& grid, variable_t const& var, std::string base_filename)
{
    std::cout << std::fixed;

    std::string cellfile = base_filename + ".cells";
    std::string scalarfile = base_filename + ".scalars";

    std::ofstream cell_file(cellfile);
    std::ofstream scalar_file(scalarfile);

    // Length of the sides of the bounding box
    double len_total[3] = {
        grid.bnd_box[0].max.x - grid.bnd_box[0].min.x,
        grid.bnd_box[0].max.y - grid.bnd_box[0].min.y,
        grid.bnd_box[0].max.z - grid.bnd_box[0].min.z
        };

    //std::cout << len_total[0] << ' ' << len_total[1] << ' ' << len_total[2] << '\n';

    int max_level = 0;
    double len[3];
    for (size_t i = 0; i < var.global_num_blocks; ++i)
    {
        if (grid.refine_level[i] > max_level)
        {
            max_level = grid.refine_level[i];
            len[0] = grid.bnd_box[i].max.x - grid.bnd_box[i].min.x;
            len[1] = grid.bnd_box[i].max.y - grid.bnd_box[i].min.y;
            len[2] = grid.bnd_box[i].max.z - grid.bnd_box[i].min.z;
        }
    }

    len[0] /= var.nxb;
    len[1] /= var.nyb;
    len[2] /= var.nzb;

    // This is the number of cells for the finest level (?)
    int vox[3];
    vox[0] = static_cast<int>(round(len_total[0] / len[0]));
    vox[1] = static_cast<int>(round(len_total[1] / len[1]));
    vox[2] = static_cast<int>(round(len_total[2] / len[2]));

    std::cout << vox[0] << ' ' << vox[1] << ' ' << vox[2] << '\n';

    //std::cout << grid.bnd_box[0].min.x << ' ' << grid.bnd_box[0].min.y << ' ' << grid.bnd_box[0].min.z << '\n';
    //std::cout << grid.bnd_box[0].max.x << ' ' << grid.bnd_box[0].max.y << ' ' << grid.bnd_box[0].max.z << '\n';
    float max_scalar = -FLT_MAX;
    float min_scalar =  FLT_MAX;
    for (size_t i = 0; i < var.global_num_blocks; ++i)
    {
        if (grid.node_type[i] == 1) // leaf!
        {
            // Project min on vox grid
            int level = max_level-grid.refine_level[i];
            int cellsize = 1<<level;

            int lower[3] = {
                static_cast<int>(round((grid.bnd_box[i].min.x - grid.bnd_box[0].min.x) / len_total[0] * vox[0])),
                static_cast<int>(round((grid.bnd_box[i].min.y - grid.bnd_box[0].min.y) / len_total[1] * vox[1])),
                static_cast<int>(round((grid.bnd_box[i].min.z - grid.bnd_box[0].min.z) / len_total[2] * vox[2]))
                };
            //std::cout << lower[0] << ' ' << lower[1] << ' ' << lower[2] << '\n';

            for (int z = 0; z < var.nzb; ++z)
            {
                for (int y = 0; y < var.nyb; ++y)
                {
                    for (int x = 0; x < var.nxb; ++x)
                    {
                        double coord[3] = {
                            (lower[0] + x*cellsize) / static_cast<double>(vox[0]) * len_total[0] + grid.bnd_box[0].min.x,
                            (lower[1] + y*cellsize) / static_cast<double>(vox[1]) * len_total[1] + grid.bnd_box[1].min.y,
                            (lower[2] + z*cellsize) / static_cast<double>(vox[2]) * len_total[2] + grid.bnd_box[2].min.z
                            };

                        //// Clip out a region of interest
                        //// Clip planes are specific to the SILCC molecular cloud data set
                        //static const double XMIN = 3.5e20;
                        //static const double XMAX = 6.2e20;
                        //static const double YMIN = -4.9e20;
                        //static const double YMAX = -2.2e20;
                        //static const double ZMIN = -12.e19;
                        //static const double ZMAX = 12.e19;
                        //if (coord[0] <  XMIN || coord[0] > XMAX || coord[1] < YMIN || coord[1] > YMAX || coord[2] < ZMIN || coord[2] > ZMAX)
                        //    continue;

                        size_t index = i * var.nxb * var.nyb * var.nzb
                                            + z * var.nyb * var.nxb
                                            + y * var.nxb
                                            + x;

                        double scalar = static_cast<double>(var.data[index]);
                        //scalar += 15429999982016076972032.000000;
                        //scalar = scalar != .0 ? log10(scalar) : scalar;
                        float scalarf = static_cast<float>(scalar);
                        max_scalar = std::max(max_scalar, scalarf);
                        min_scalar = std::min(min_scalar, scalarf);
                        int lowerer[3] = { lower[0] + x*cellsize, lower[1] + y*cellsize, lower[2] + z*cellsize };
                        cell_file.write((char*)&lowerer, sizeof(int32_t) * 3);
                        cell_file.write((char*)&level, sizeof(int32_t));
                        scalar_file.write((char*)&scalarf, sizeof(float));
                    }
                }
            }
        }
    }
    std::cout << min_scalar << ' ' << max_scalar << '\n';
    std::cout << "Output written to:\n";
    std::cout << "Scalars: " << scalarfile << '\n';
    std::cout << "Cells:   " << cellfile << '\n';
}

int main(int argc, char** argv)
{
    std::string filename;
    std::string outname;
    std::string var;
    bool list_mode = false;

    cl::Cmdline cli("exaFlashToCells", "Convert FLASH4 HDF5 files (http://flash.uchicago.edu/site/) to cells");

    cli.Add(
        "filename",
        "filename (FLASH4 format)",
        cl::Required::yes | cl::Arg::yes | cl::Positional::yes,
        cl::Var(filename)
        );

    cli.Add(
        "o",
        "Output base filename (default: <filename>_<var>)",
        cl::Required::no | cl::Arg::required,
        cl::Var(outname)
        );

    cli.Add(
        "v|var",
        "Load variable <arg>. If not specified, list all variables contained in <filename> and exit.",
        cl::Required::no | cl::Arg::required,
        cl::Var(var)
        );

    cli.Add(
        "l|list",
        "List all variables contained in <filename> and exit",
        cl::Required::no,
        cl::Var(list_mode)
        );

    auto result = cli.Parse(argv+1,argv+argc);

    cli.PrintDiag();

    if (!result)
    {
        cli.PrintHelp();
        return EXIT_FAILURE;
    }

    // option --var not specified, just list the available variables
    if (var.empty())
    {
        cli.PrintHelp();
        std::cout << '\n';
        list_mode = true;
    }

    if (outname.empty())
    {
        outname = filename + "_" + var;
    }

    try
    {
        H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);
        //H5::H5File file("SILCC_hdf5_plt_cnt_0300", H5F_ACC_RDONLY);

        // Read simulation info
        sim_info_t sim_info;
        read_sim_info(sim_info, file);

        // Read grid data
        grid_t grid;
        read_grid(grid, file);

        if (list_mode)
        {
            // Don't convert but just list all the 4char variable
            // names contained in the file
            std::cout << "Variable names that can be passed to option --var:\n";
            for (std::size_t i = 0; i < grid.unknown_names.size(); ++i)
            {
                std::string uname(grid.unknown_names[i].data(),grid.unknown_names[i].data()+4);
                std::cout << uname << '\n';
            }
            return EXIT_SUCCESS;
        }
        else
        {
            // Read data
            variable_t density;
            read_variable(density, file, var.c_str());

            make_cells(grid, density, outname);
        }
    }
    catch (H5::FileIException error)
    {
        error.printError();
        return EXIT_FAILURE;
    }
    catch (H5::DataSpaceIException error)
    {
        error.printError();
        return EXIT_FAILURE;
    }
    catch (H5::DataTypeIException error)
    {
        error.printError();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
