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

#include <cassert>
#include <iostream>
#include <stdexcept>
#include "Texture.h"

namespace exa {

  Texture::Texture(int deviceCount,
                   int dims,
                   int x,
                   int y,
                   int z,
                   int w,
                   cudaChannelFormatKind f)
  {
    reset(deviceCount,dims,x,y,z,w,f);
  }

  Texture::~Texture()
  {
    int prevDeviceID = -1;
    cudaGetDevice(&prevDeviceID);
    for (int deviceID=0;deviceID<perDevice.size();deviceID++) {
      cudaSetDevice(deviceID);
      cudaDestroyTextureObject(perDevice[deviceID].textureObject);
    }
    cudaSetDevice(prevDeviceID);
  }
  
  void Texture::reset(int deviceCount, int dims, int x, int y, int z, int w, cudaChannelFormatKind f)
  {
    assert(dims>=1 && dims<=3);
    this->dims = dims;
    perDevice.resize(deviceCount);
    desc = cudaCreateChannelDesc(x,y,z,w,f);
  }

  void Texture::resize(int width, int height, int depth)
  {
    this->width  = width;
    this->height = height;
    this->depth  = depth;

    int prevDeviceID = -1;
    cudaGetDevice(&prevDeviceID);
    for (int deviceID=0;deviceID<perDevice.size();deviceID++) {
      cudaSetDevice(deviceID);

      cudaError_t err = cudaSuccess;

      switch (dims) {
      case 1:
        err = perDevice[deviceID].cudaArray.allocate(desc,width);
        break;
      case 3:
        err = perDevice[deviceID].cudaArray.allocate3D(desc,width,height,depth);
        break;
      default:
        throw std::runtime_error("Unsupported texture dimensions");
      }

      if (err != cudaSuccess)
        throw std::runtime_error("Error allocating texture memory");
    }

    cudaSetDevice(prevDeviceID);

    resetTextureObjects();
  }

  void Texture::resetTextureObjects()
  {
    int prevDeviceID = -1;
    cudaGetDevice(&prevDeviceID);
    for (int deviceID=0;deviceID<perDevice.size();deviceID++) {
      cudaSetDevice(deviceID);

      cudaResourceDesc resource_desc;
      memset(&resource_desc, 0, sizeof(resource_desc));
      resource_desc.resType                   = cudaResourceTypeArray;
      resource_desc.res.array.array           = perDevice[deviceID].cudaArray.get();

      cudaTextureDesc texture_desc;
      memset(&texture_desc, 0, sizeof(texture_desc));
      texture_desc.addressMode[0]             = addressMode[0];
      texture_desc.addressMode[1]             = addressMode[1];
      texture_desc.addressMode[2]             = addressMode[2];
      texture_desc.addressMode[3]             = addressMode[3];
      texture_desc.filterMode                 = filterMode;
      texture_desc.readMode                   = readMode;
      texture_desc.sRGB                       = sRGB;
      texture_desc.normalizedCoords           = normalizedCoords;

      if (perDevice[deviceID].textureObject)
        cudaDestroyTextureObject(perDevice[deviceID].textureObject);

      cudaError_t err = cudaCreateTextureObject(&perDevice[deviceID].textureObject,
                                                &resource_desc, &texture_desc, 0);

      if (err != cudaSuccess)
        throw std::runtime_error("Error creating texture objects");
    }
    cudaSetDevice(prevDeviceID);
  }

} // ::optix
