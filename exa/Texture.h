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

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "cuda/Array.h"

namespace exa {

  struct Texture
  {
    typedef std::shared_ptr<Texture> SP;

    Texture() = default;
    Texture(int deviceCount,
            int dims,
            int x,
            int y,
            int z,
            int w,
            cudaChannelFormatKind f);
   ~Texture();

    cudaChannelFormatDesc desc;

    void reset(int deviceCount, int dims, int x, int y, int z, int w, cudaChannelFormatKind f);

    void resize(int width, int height=1, int depth=1);

    cudaTextureObject_t& get(int deviceID) { return perDevice[deviceID].textureObject; }
    const cudaTextureObject_t& get(int deviceID) const { return perDevice[deviceID].textureObject; }

    template<typename T>
    bool upload(int deviceID, const T *ptr) {
      int prevDeviceID = -1;
      cudaGetDevice(&prevDeviceID);
      cudaSetDevice(deviceID);

      cudaError_t err = cudaSuccess;

      switch (dims)
      {
      case 1:
        err = perDevice[deviceID].cudaArray.upload(ptr,width*sizeof(T));
        break;
      case 3:
        err = perDevice[deviceID].cudaArray.upload(ptr,width,height,depth);
        break;
      default:
        cudaSetDevice(prevDeviceID);
        return false;
      }

      cudaSetDevice(prevDeviceID);
      return err == cudaSuccess;
    }

    template<typename T>
    bool upload(const T *ptr) {
      bool success=true;
      for (int deviceID=0;deviceID<perDevice.size();deviceID++) {

        if (!upload(deviceID,ptr))
          success=false;

        if (!success)
          break;
      }

      return success;
    }

    template<typename T>
    bool uploadAsync(int deviceID, cudaStream_t stream, const T *ptr) {
      int prevDeviceID = -1;
      cudaGetDevice(&prevDeviceID);
      cudaSetDevice(deviceID);

      switch (dims)
      {
      case 1:
        throw std::runtime_error("Not implemented yet");
        perDevice[deviceID].cudaArray.uploadAsync(stream,ptr,width*sizeof(T));
        //break;
      case 3:
        perDevice[deviceID].cudaArray.uploadAsync(stream,ptr,width,height,depth);
        break;
      default:
        cudaSetDevice(prevDeviceID);
        return false;
      }

      cudaSetDevice(prevDeviceID);
      return true;
    }

    void setAddressMode(cudaTextureAddressMode am) {
      addressMode[0]=addressMode[1]=addressMode[2]=addressMode[3]=am;
      resetTextureObjects();
    }

    void setFilterMode(cudaTextureFilterMode fm) {
      filterMode=fm;
      resetTextureObjects();
    }

    void setReadMode(cudaTextureReadMode rm) {
      readMode=rm;
      resetTextureObjects();
    }

    void setSRGB(bool srgb) {
      sRGB=srgb;
      resetTextureObjects();
    }

    void setNormalizedCoords(bool nc) {
      normalizedCoords=nc;
      resetTextureObjects();
    }

  private:
    int dims;
    int width=0, height=0, depth=0;
    cudaTextureAddressMode addressMode[4] = { cudaAddressModeClamp };
    cudaTextureFilterMode filterMode = cudaFilterModeLinear;
    cudaTextureReadMode readMode = cudaReadModeElementType;
    bool sRGB = false;
    bool normalizedCoords = true;

    void resetTextureObjects();

    struct PerDevice {
      cudaTextureObject_t textureObject;
      cuda::Array cudaArray;
    };
    std::vector<PerDevice> perDevice;

  };
  
}

