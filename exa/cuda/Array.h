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

#include <cstddef>
#include <cstring> // memset

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace cuda
{
  struct Array
  {
    // width and height are always *elements*

    Array() = default;

    Array(cudaChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
      allocate(desc, width, height, flags);
    }

    Array(cudaChannelFormatDesc const& desc, size_t width, size_t height, size_t depth, unsigned flags)
    {
      allocate3D(desc, width, height, depth, flags);
    }

    Array(Array&& rhs)
      : arrayPtr_(rhs.release())
    {
    }

   ~Array()
    {
      reset();
    }

    Array& operator=(Array&& rhs)
    {
      reset( rhs.release() );
      return *this;
    }


    // NOT copyable
    Array(const Array &rhs) = delete;
    Array& operator=(const Array &rhs) = delete;


    cudaArray_t get() const
    {
      return arrayPtr_;
    }


    cudaError_t allocate(cudaChannelFormatDesc const& desc, size_t width, size_t height = 0, unsigned flags = 0)
    {
      cudaFree(arrayPtr_);

      cudaError_t err = cudaMallocArray(&arrayPtr_,
                                        &desc,
                                        width,
                                        height,
                                        flags);

      if (err != cudaSuccess)
      {
        arrayPtr_ = nullptr;
      }

      return err;
    }

    cudaError_t allocate3D(cudaChannelFormatDesc const& desc, size_t width, size_t height, size_t depth, unsigned int flags = 0)
    {
      cudaFree(arrayPtr_);

      cudaExtent extent { width, height, depth };

      cudaError_t err = cudaMalloc3DArray(&arrayPtr_, &desc, extent, flags);

      if (err != cudaSuccess)
      {
          arrayPtr_ = nullptr;
      }

      return err;
    }

    template<typename T>
    cudaError_t upload(T const* host_data, size_t count)
    {
      return cudaMemcpy2DToArray(arrayPtr_,
                                 0,
                                 0,
                                 host_data,
                                 count,
                                 count,
                                 1,
                                 cudaMemcpyHostToDevice);
    }

    template<typename T>
    cudaError_t upload(T const* host_data, size_t width, size_t height, size_t depth)
    {
      cudaMemcpy3DParms copy_params;
      memset(&copy_params, 0, sizeof(copy_params));
      copy_params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(host_data),
                                               width * sizeof(T),
                                               width,
                                               height);

      copy_params.dstArray = arrayPtr_;
      copy_params.extent   = { width, height, depth };
      copy_params.kind     = cudaMemcpyHostToDevice;

      return cudaMemcpy3D(&copy_params);
    }

    template<typename T>
    cudaError_t uploadAsync(cudaStream_t stream, T const* host_data, size_t width, size_t height, size_t depth)
    {
      cudaMemcpy3DParms copy_params;
      memset(&copy_params, 0, sizeof(copy_params));
      copy_params.srcPtr = make_cudaPitchedPtr(const_cast<T*>(host_data),
                                               width * sizeof(T),
                                               width,
                                               height);

      copy_params.dstArray = arrayPtr_;
      copy_params.extent   = { width, height, depth };
      copy_params.kind     = cudaMemcpyHostToDevice;

      return cudaMemcpy3DAsync(&copy_params,stream);
    }

  private:
    cudaArray_t arrayPtr_  = nullptr;

    cudaArray_t release()
    {
      cudaArray_t ptr = arrayPtr_;
      arrayPtr_ = nullptr;
      return ptr;
    }

    void reset(cudaArray_t ptr = nullptr)
    {
      if (arrayPtr_)
      {
        cudaFree(arrayPtr_);
      }

      arrayPtr_ = ptr;
    }
  };

} // ::cuda

