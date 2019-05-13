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

#include <random>
#include <float.h>
#include <iomanip>
#include <numeric>
#include <sstream>
#include "exa/OptixRenderer.h"
#include "programs/FrameState.h"
#include "programs/Brick.h"
#include "programs/IsoSurfaceData.h"
#include "programs/LaunchParams.h"
#include "programs/StreamlineData.h"
#include "programs/SurfaceGeomData.h"
#include "programs/VolumeData.h"

extern "C" const char ptxCode[];

namespace exa {

  using std::min;
  using std::max;

  OptixRenderer::OptixRenderer(ExaBricks::SP input,
                               std::vector<TriangleMesh::SP> surfaces,
                               std::vector<ScalarField::SP>  scalarFields)
    : input(input), scalarFields(scalarFields), surfaces(surfaces)
  {
    voxelSpaceBounds = input->getBounds();
    
    context = owlContextCreate();
    module = owlModuleCreate(context, ptxCode);

    // -------------------------------------------------------
    // create frame state buffer
    // -------------------------------------------------------
    frameStateBuffer = owlDeviceBufferCreate(context,
                                             OWL_USER_TYPE(FrameState),
                                             1,nullptr);

    // -------------------------------------------------------
    // transfer function textures
    // -------------------------------------------------------
    for (int c=0; c<MAX_CHANNELS; ++c) {
      transFunc[c].reset(owlGetDeviceCount(context),1,32,32,32,32,cudaChannelFormatKindFloat);
      transFunc[c].resize(NUM_XF_VALUES);
    }

    // -------------------------------------------------------
    // create brick and index buffers
    // -------------------------------------------------------
    std::cout << "Bricks Memory: "
        << owl::prettyNumber(input->bricks.size() * sizeof(Brick)) << " bytes\n";

    std::cout << "creating index vector, total num cells: "
              << prettyDouble(input->totalNumCells) << std::endl;
    indexVector.resize(input->totalNumCells);
    
    std::vector<Brick> brickVector(input->bricks.size());
    Brick *brick = brickVector.data();
    int *index = (int *)indexVector.data();
    size_t scalarOffset = 0;
    for (int brickID=0;brickID<input->bricks.size();brickID++) {
      brick[brickID].lower = input->bricks[brickID]->lower;
      brick[brickID].size  = input->bricks[brickID]->size;
      brick[brickID].level = input->bricks[brickID]->level;
      brick[brickID].begin = (int)scalarOffset;
      if (brick[brickID].begin < 0)
        throw std::runtime_error("32-bit offset overflow");

      std::copy(input->bricks[brickID]->cellIDs.begin(),
                input->bricks[brickID]->cellIDs.end(),
                index + scalarOffset);
      scalarOffset += volume(brick[brickID].size);
      if (input->bricks[brickID]->cellIDs.size() != volume(brick[brickID].size))
        throw std::runtime_error("failed sanity-check in brick size");
      if ((size_t(brick[brickID].begin) + volume(brick[brickID].size)) != scalarOffset)
        throw std::runtime_error("overflow in 32-bit brick offset");
    }

    brickBuffer = owlDeviceBufferCreate(context,
                                        OWL_USER_TYPE(Brick),
                                        brickVector.size(),
                                        brickVector.data());

    // -------------------------------------------------------
    // create scalar buffer
    // -------------------------------------------------------
    size_t scalarsMemory = 0;
    size_t scalarsOffset = 0;
    std::vector<float> scalarBuffersVector(indexVector.size() * scalarFields.size());
    std::vector<unsigned> offsetsVector(scalarFields.size());
    unsigned *offsets = offsetsVector.data();
    for (const auto &sf : scalarFields) {
      float *scalar = scalarBuffersVector.data() + scalarsOffset;
      *offsets = scalarsOffset;
      ++offsets;
      scalarsMemory += indexVector.size() * sizeof(float);
      scalarsOffset += indexVector.size();
      parallel_for_blocked(0ull,indexVector.size(),1024*1024,[&](size_t begin,size_t end){
          for (size_t i=begin;i<end;i++) {
            if (indexVector[i] < 0) {
#if ALLOW_EMPTY_CELLS
              scalar[i] = EMPTY_CELL_POISON_VALUE;
#else
              throw std::runtime_error("overflow in index vector...");
#endif
            } else {
              int cellID = indexVector[i];
              if (cellID < 0)
                throw std::runtime_error("negative cell ID");
              if (cellID >= sf->value.size())
                throw std::runtime_error("invalid cell ID");
              scalar[i] = sf->value[cellID];
            }
          }
        });
    }
    scalarBuffers = owlDeviceBufferCreate(context,
                                          OWL_FLOAT,
                                          scalarBuffersVector.size(),
                                          scalarBuffersVector.data());

    scalarBufferOffsets = owlDeviceBufferCreate(context,
                                                OWL_UINT,
                                                offsetsVector.size(),
                                                offsetsVector.data());
    std::cout << "# Scalars: " << scalarFields.size()
        << "\nScalars Memory: " << owl::prettyNumber(scalarsMemory) << " bytes\n";

#if EXPLICIT_BASIS_METHOD
    // -------------------------------------------------------
    // create regions
    // -------------------------------------------------------
    // Build for the primary field (isosurface/vol render field)
    // NOTE: This would have to be rebuilt if the primary field is changed
    {
      float *scalar = scalarBuffersVector.data();
      unsigned *offsets = offsetsVector.data();
      size_t numScalarFields = multiFieldDvr? scalarFields.size(): 1;
      regions.buildFrom(input,// validBricks,
                        brick,input->bricks.size(),scalar,offsets,numScalarFields);
    }

    std::cout << "uploading same-brick regions" << std::endl;
    regionsBuffer = owlDeviceBufferCreate(context,
                                          OWL_USER_TYPE(SameBricksRegion),
                                          regions.brickRegions.size(), //sf->value.size());
                                          regions.brickRegions.data());

    regionsLeafListBuffer = owlDeviceBufferCreate(context,
                                                  OWL_INT,
                                                  regions.leafList.size(), //sf->value.size());
                                                  regions.leafList.data());

    size_t regionsMemory = regions.brickRegions.size() * sizeof(SameBricksRegion)
        + regions.leafList.size() * sizeof(int);
    std::cout << "Regions Memory: " << owl::prettyNumber(regionsMemory) << " bytes\n";


#else
    // Determine value ranges per brick for space skipping
    // Might not be the fastest way to do this..
    std::vector<range1f> valueRangePerBrick(brickVector.size());
    //for (size_t f=0; f<scalarFields.size(); ++f) {
    { size_t f = 0; // only for the first field
      for (size_t brickID=0; brickID<brickVector.size(); ++brickID) {
        const int brickIdx = f*brickVector.size()+brickID;
        const int cellIdx = brick[brickID].begin;
        const int numCells = volume(brick[brickID].size);

        const size_t first(cellIdx);
        const size_t last = first + numCells;

        valueRangePerBrick[brickID] = range1f(FLT_MAX,-FLT_MAX);
        for (size_t i = first; i < last; ++i) {
          float value = scalarFields[f]->value[i];
          valueRangePerBrick[brickID].lower = fminf(valueRangePerBrick[brickID].lower,value);
          valueRangePerBrick[brickID].upper = fmaxf(valueRangePerBrick[brickID].upper,value);
        }
      }
    }

    valueRangePerBrickBuffer = owlDeviceBufferCreate(context,
                                                     OWL_USER_TYPE(range1f),
                                                     valueRangePerBrick.size(),
                                                     valueRangePerBrick.data());
#endif
    
    std::vector<int> primaryChannels(multiFieldDvr? scalarFields.size(): 1);
    std::iota(primaryChannels.begin(),primaryChannels.end(),0);//0,1,2,...
    primaryChannelsBuffer = owlDeviceBufferCreate(context,
                                                  OWL_INT,
                                                  primaryChannels.size(),
                                                  primaryChannels.data());

    // tracer

    // Generate random seeds inside box
    traces.buffer = owlDeviceBufferCreate(context,
                                          OWL_FLOAT3,
                                          0, nullptr);
    traces.currentTimestep = owlDeviceBufferCreate(context,OWL_INT,1,&traces.timestepHost);

    // -------------------------------------------------------
    // set up empty user geom miss progs
    // -------------------------------------------------------
    OWLMissProg missProg
      = owlMissProgCreate(context,module,"missProg",0,
                          nullptr,-1);

    // -------------------------------------------------------
    // set up launch params
    // -------------------------------------------------------
    OWLVarDecl launchParamsVars[] = {
      // Render state
      { "deviceIndex",              OWL_DEVICE, OWL_OFFSETOF(LaunchParams,deviceIndex)},
      { "deviceCount",              OWL_INT,    OWL_OFFSETOF(LaunchParams,deviceCount)},
      { "volumeBVH",                OWL_GROUP,  OWL_OFFSETOF(LaunchParams,volumeBVH)},
      { "surfaceModel",             OWL_GROUP,  OWL_OFFSETOF(LaunchParams,surfaceModel)},
      { "isoSurfaceBVH",            OWL_GROUP,  OWL_OFFSETOF(LaunchParams,isoSurfaceBVH)},
      { "streamlineBVH",            OWL_GROUP,  OWL_OFFSETOF(LaunchParams,streamlineBVH)},
      { "fbSize",                   OWL_INT2,   OWL_OFFSETOF(LaunchParams,fbSize)},
      { "worldSpaceBounds_lo",      OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,worldSpaceBounds_lo)},
      { "worldSpaceBounds_hi",      OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,worldSpaceBounds_hi)},
      { "voxelSpaceBounds_lo",      OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,voxelSpaceBounds_lo)},
      { "voxelSpaceBounds_hi",      OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,voxelSpaceBounds_hi)},
      { "colorBufferPtr",           OWL_RAW_POINTER,OWL_OFFSETOF(LaunchParams,colorBufferPtr)},
      { "accumBufferPtr",           OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBufferPtr)},
      { "frameStateBuffer",         OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,frameStateBuffer)},
      { "dt",                       OWL_FLOAT,  OWL_OFFSETOF(LaunchParams,dt)},
      { "sameBrickRegionsBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,sameBrickRegionsBuffer)},
      { "sameBrickRegionsLeafList", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,sameBrickRegionsLeafList)},
      { "brickBuffer",              OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,brickBuffer)},
      { "scalarBuffers",            OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,scalarBuffers)},
      { "scalarBufferOffsets",      OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,scalarBufferOffsets)},
      { "primaryChannels",          OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,primaryChannels)},
      { "colormapChannel",          OWL_INT,    OWL_OFFSETOF(LaunchParams,colormapChannel)},
      { "numPrimaryChannels",       OWL_INT,    OWL_OFFSETOF(LaunchParams,numPrimaryChannels)},
      { "gradientShadingDVR",       OWL_INT,    OWL_OFFSETOF(LaunchParams,gradientShadingDVR)},
      { "gradientShadingISO",       OWL_INT,    OWL_OFFSETOF(LaunchParams,gradientShadingISO)},
      // tracer
      { "tracerEnabled",            OWL_INT,    OWL_OFFSETOF(LaunchParams,tracerEnabled)},
      { "traces",                   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,traces)},
      { "tracerChannels",           OWL_INT3,   OWL_OFFSETOF(LaunchParams,tracerChannels)},
      { "numTraces",                OWL_INT,    OWL_OFFSETOF(LaunchParams,numTraces)},
      { "numTimesteps",             OWL_INT,    OWL_OFFSETOF(LaunchParams,numTimesteps)},
      { "steplen",                  OWL_FLOAT,  OWL_OFFSETOF(LaunchParams,steplen)},
      { "currentTimestep",          OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,currentTimestep)},
      // Debugging
      { /* sentinel to mark end of list */ }
    };
    launchParams
      = owlParamsCreate(context,sizeof(LaunchParams),
                        launchParamsVars,-1);

    owlParamsSet3f(launchParams,"worldSpaceBounds_lo",(const owl3f&)worldSpaceBounds.lower);
    owlParamsSet3f(launchParams,"worldSpaceBounds_hi",(const owl3f&)worldSpaceBounds.upper);
    owlParamsSet3f(launchParams,"voxelSpaceBounds_lo",(const owl3f&)voxelSpaceBounds.lower);
    owlParamsSet3f(launchParams,"voxelSpaceBounds_hi",(const owl3f&)voxelSpaceBounds.upper);

    owlParamsSetBuffer(launchParams,"primaryChannels",primaryChannelsBuffer);
    if (!multiFieldDvr && scalarFields.size() > 1) {
      std::cout << "using second channel to colormap\n";
      owlParamsSet1i(launchParams,"colormapChannel",1);
    } else {
      owlParamsSet1i(launchParams,"colormapChannel",0);
    }
    owlParamsSet1i(launchParams,"numPrimaryChannels",multiFieldDvr? scalarFields.size(): 1);
    owlParamsSet1i(launchParams,"gradientShadingDVR",gradientShadingDVR);
    owlParamsSet1i(launchParams,"gradientShadingISO",gradientShadingISO);

    owlParamsSetBuffer(launchParams,"frameStateBuffer",frameStateBuffer);    

#if EXPLICIT_BASIS_METHOD
    owlParamsSetBuffer(launchParams,"sameBrickRegionsBuffer",regionsBuffer);
    owlParamsSetBuffer(launchParams,"sameBrickRegionsLeafList",regionsLeafListBuffer);
#endif
    owlParamsSetBuffer(launchParams,"brickBuffer",brickBuffer);
    resetTracer();
    owlParamsSetBuffer(launchParams,"scalarBuffers",scalarBuffers);
    owlParamsSetBuffer(launchParams,"scalarBufferOffsets",scalarBufferOffsets);

    raygenProgram =
        owlRayGenCreate(context,module,"renderFrame",
                        /* no sbt data: */0,nullptr,-1);

    // -------------------------------------------------------
    // create stuff
    // -------------------------------------------------------
    createSurfaces();
      
    createVolumeBVH();

    createIsoSurfaceBVH();

    createStreamlineBVH();

    owlBuildPrograms(context);
    owlBuildPipeline(context);

    owlBuildSBT(context);
  }

  OptixRenderer::~OptixRenderer()
  { /* empty */ }
    
  void OptixRenderer::setVoxelSpaceTransform(const affine3f &voxelSpaceTransform)
  {
    FrameState *fs = (FrameState *)&frameState;
    // frameStateBuffer->map();
    fs->voxelSpaceTransform = voxelSpaceTransform;
    // frameStateBuffer->unmap();
    owlBufferUpload(frameStateBuffer,fs);

    worldSpaceBounds
      = box3f(xfmPoint(rcp(fs->voxelSpaceTransform),voxelSpaceBounds.lower),
              xfmPoint(rcp(fs->voxelSpaceTransform),voxelSpaceBounds.upper));

    owlParamsSet3f(launchParams,"worldSpaceBounds_lo",(const owl3f&)worldSpaceBounds.lower);
    owlParamsSet3f(launchParams,"worldSpaceBounds_hi",(const owl3f&)worldSpaceBounds.upper);

    owlParamsSet3f(launchParams,"voxelSpaceBounds_lo",(const owl3f&)voxelSpaceBounds.lower);
    owlParamsSet3f(launchParams,"voxelSpaceBounds_hi",(const owl3f&)voxelSpaceBounds.upper);
  }
     
  void OptixRenderer::resizeFrameBuffer(void *fbPointer, const vec2i &fbSize)
  {
    this->fbSize = fbSize;
    if (!accumBuffer)
      accumBuffer = owlDeviceBufferCreate(context,OWL_FLOAT4,fbSize.x*fbSize.y,nullptr);
    owlBufferResize(accumBuffer,fbSize.x*fbSize.y);
    owlParamsSetBuffer(launchParams,"accumBufferPtr",accumBuffer);
    owlParamsSet1i(launchParams,"deviceCount",owlGetDeviceCount(context));
      
    if (!colorBuffer)
      colorBuffer = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);
    owlBufferResize(colorBuffer,fbSize.x*fbSize.y);
    owlParamsSet1ul(launchParams,"colorBufferPtr",(uint64_t)fbPointer);
    owlParamsSet2i(launchParams,"fbSize",fbSize.x,fbSize.y);
  }

  void OptixRenderer::updateCamera(const vec3f &pos,
                                   const vec3f &dir00,
                                   const vec3f &dirDu,
                                   const vec3f &dirDv)
  {
    FrameState *fs = (FrameState *)&frameState;
    fs->camera.pos   = pos;
    fs->camera.dir00 = dir00;
    fs->camera.dirDu = dirDu;
    fs->camera.dirDv = dirDv;
    owlBufferUpload(frameStateBuffer,fs);
  }

  void OptixRenderer::updateXF(int chan,
                               const float *opacities,
                               const std::vector<vec3f>  &colorMap,
                               const interval<float>     &xfDomain,
                               /*! the value we'll scale the final
                                   opacity value of each sample with -
                                   allows for making volume more
                                   transparent than a 0/1 range for
                                   each sample would indicate */
                               float xfOpacityScale)
  {
    FrameState *fs = (FrameState *)&frameState;
    if (colorMap.size() != NUM_XF_VALUES)
      throw std::runtime_error("mismatching xf size!?");
    fs->xfDomain[chan] = xfDomain;
    fs->xfOpacityScale = xfOpacityScale;

    std::vector<vec4f> lut(NUM_XF_VALUES);
    for (int i=0;i<NUM_XF_VALUES;i++) {
      lut[i] = vec4f(colorMap[i],opacities[i]);
    }
    transFunc[chan].upload(lut.data());

    int prevDeviceID = -1;
    cudaGetDevice(&prevDeviceID);
    for (int deviceID=0; deviceID != owlGetDeviceCount(context); ++deviceID) {
      cudaSetDevice(deviceID);
      fs->xfTexture[chan] = transFunc[chan].get(deviceID);
      cudaMemcpy((void*)owlBufferGetPointer(frameStateBuffer,deviceID),
                 fs, sizeof(*fs), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prevDeviceID);

    needVolumeBVHRebuild = true;
  }

  void OptixRenderer::updateFrameID(int frameID)
  {
    FrameState *fs = (FrameState *)&frameState;
    fs->frameID = frameID;
    owlBufferUpload(frameStateBuffer,fs);
  }

  void OptixRenderer::updateDt(float dt)
  {
    owlParamsSet1f(launchParams,"dt",dt);
  }

  void OptixRenderer::setSpaceSkipping(bool enable)
  {
    FrameState *fs = (FrameState *)&frameState;
    bool contourPlanesActive = false;
    for (int i=0; i<MAX_CONTOUR_PLANES; ++i) {
      if (fs->contourPlane[i].enabled) {
        contourPlanesActive = true;
        break;
      }
    }
    doSpaceSkipping = enable;
    owlGeomSet1i(volumeGeom,"spaceSkippingEnabled",
                 !contourPlanesActive && doSpaceSkipping);
    needVolumeBVHRebuild = true;
  }

  void OptixRenderer::setGradientShadingDVR(bool enable)
  {
    owlParamsSet1i(launchParams,"gradientShadingDVR",(int)enable);
  }

  void OptixRenderer::setGradientShadingISO(bool enable)
  {
    owlParamsSet1i(launchParams,"gradientShadingISO",(int)enable);
  }

  void OptixRenderer::setTracerEnabled(bool enable)
  {
    traces.tracerEnabled = enable;
    owlParamsSet1i(launchParams,"tracerEnabled",traces.tracerEnabled);
  }

  void OptixRenderer::resetTracer()
  {
    std::vector<vec3f> hostTraces(traces.numTraces*traces.numTimesteps);
    vec3f size = voxelSpaceBounds.upper-voxelSpaceBounds.lower;
    std::default_random_engine engine(0);
    std::uniform_real_distribution<float> x(traces.seedRegion.lower.x*size.x,traces.seedRegion.upper.x*size.x);
    std::uniform_real_distribution<float> y(traces.seedRegion.lower.y*size.y,traces.seedRegion.upper.y*size.y);
    std::uniform_real_distribution<float> z(traces.seedRegion.lower.z*size.z,traces.seedRegion.upper.z*size.z);
    for (int i=0; i<traces.numTraces; ++i) {
      hostTraces[i*traces.numTimesteps] = vec3f(x(engine),y(engine),z(engine));
    }
    owlBufferResize(traces.buffer,hostTraces.size());
    owlBufferUpload(traces.buffer,hostTraces.data());
    traces.timestepHost=0;
    owlBufferUpload(traces.currentTimestep,&traces.timestepHost);
    needStreamlineBVHRebuild = true;

    owlParamsSet1i(launchParams,"tracerEnabled",traces.tracerEnabled);
    owlParamsSetBuffer(launchParams,"traces",traces.buffer);
    owlParamsSet3i(launchParams,"tracerChannels",traces.tracerChannels.x,traces.tracerChannels.y,traces.tracerChannels.z);
    owlParamsSet1i(launchParams,"numTraces",traces.numTraces);
    owlParamsSet1i(launchParams,"numTimesteps",traces.numTimesteps);
    owlParamsSet1f(launchParams,"steplen",traces.steplen);
    owlParamsSetBuffer(launchParams,"currentTimestep",traces.currentTimestep);
  }

  bool OptixRenderer::advanceTracer()
  {
    if (!traces.tracerEnabled)
      return false;

    traces.timestepHost++;
    if (traces.timestepHost <= traces.numTimesteps) {
      owlBufferUpload(traces.currentTimestep,&traces.timestepHost);
      needStreamlineBVHRebuild = true;
    }
    return needStreamlineBVHRebuild;
  }

  void OptixRenderer::updateIsoValues(const float *isoValues,
                                      const int   *channels,
                                      const int   *enabled)
  {
    FrameState *fs = (FrameState *)&frameState;
    for (int i=0;i<MAX_ISO_SURFACES;i++) {
      fs->isoSurface[i].value = isoValues[i];
      fs->isoSurface[i].channel = channels[i];
      fs->isoSurface[i].enabled = enabled[i];
    }
    owlBufferUpload(frameStateBuffer,fs);

    needIsoBVHRebuild = true;
  }

  void OptixRenderer::updateContourPlanes(const vec3f *normals,
                                          const float *offsets,
                                          const int   *channels,
                                          const int   *enabled)
  {
    FrameState *fs = (FrameState *)&frameState;
    for (int i=0;i<MAX_CONTOUR_PLANES;i++) {
      fs->contourPlane[i].normal = normalize(normals[i]);
      fs->contourPlane[i].offset = offsets[i];
      fs->contourPlane[i].channel = channels[i];
      fs->contourPlane[i].enabled = enabled[i];
    }
    owlBufferUpload(frameStateBuffer,fs);

    bool contourPlanesActive = false;
    for (int i=0; i<MAX_CONTOUR_PLANES; ++i) {
      if (fs->contourPlane[i].enabled) {
        contourPlanesActive = true;
        break;
      }
    }
    owlGeomSet1i(volumeGeom,"spaceSkippingEnabled",
                 !contourPlanesActive && doSpaceSkipping);

    needVolumeBVHRebuild = true;
  }

  void OptixRenderer::render()
  {
    if (needVolumeBVHRebuild) {
      owlGroupBuildAccel(volumeGroup);
      owlGroupBuildAccel(volumeBVH);
      needVolumeBVHRebuild = false;
    }

    if (needIsoBVHRebuild) {
      owlGroupBuildAccel(isoSurfaceGroup);
      owlGroupBuildAccel(isoSurfaceBVH);
      needIsoBVHRebuild = false;
    }

    if (needStreamlineBVHRebuild) {
      owlGroupBuildAccel(streamlineGroup);
      owlGroupBuildAccel(streamlineBVH);
      needStreamlineBVHRebuild = false;
    }

    owlLaunch2D(raygenProgram,fbSize.x,fbSize.y,launchParams);
  }

  void OptixRenderer::createSurfaces()
  {
    OWLVarDecl surfaceGeomVars[] = {
      { "indexBuffer",  OWL_BUFPTR, OWL_OFFSETOF(SurfaceGeomData,indexBuffer)},
      { "vertexBuffer", OWL_BUFPTR, OWL_OFFSETOF(SurfaceGeomData,vertexBuffer)},
      { nullptr /* sentinel to mark end of list */ }
    };
    surfaceGeomType = owlGeomTypeCreate(context,
                                        OWL_TRIANGLES,
                                        sizeof(SurfaceGeomData),
                                        surfaceGeomVars, 2);

    owlGeomTypeSetClosestHit(surfaceGeomType, 0,
                             module, "SurfaceBVH");

    if (surfaces.empty())  {
      OWLGeom surfaceGeom = owlGeomCreate(context, surfaceGeomType);
      surfaceModel = owlTrianglesGeomGroupCreate(context, 1, &surfaceGeom);
      //owlGroupBuildAccel(surfaceModel);
      return;
    }

    surfaceModel = owlInstanceGroupCreate(context,surfaces.size());
    std::cout << "#exa: number of surfaces " << prettyDouble(surfaces.size()) << std::endl;
    for (int meshID=0;meshID<surfaces.size();meshID++) {
      auto &mesh = surfaces[meshID];
      OWLGroup gi;
      OWLGeom geom;
      OWLBuffer vertexBuffer;
      OWLBuffer indexBuffer;

      indexBuffer = owlDeviceBufferCreate(context, OWL_INT3,
                                          mesh->index.size(), mesh->index.data());
      vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3,
                                           mesh->vertex.size(),mesh->vertex.data());

      geom = owlGeomCreate(context, surfaceGeomType);
      owlGeomSetBuffer(geom, "vertexBuffer", vertexBuffer);
      owlGeomSetBuffer(geom, "indexBuffer", indexBuffer);
      owlTrianglesSetIndices(geom, indexBuffer, mesh->index.size(), sizeof(vec3i), 0);
      owlTrianglesSetVertices(geom, vertexBuffer, mesh->vertex.size(), sizeof(vec3f), 0);

      gi = owlTrianglesGeomGroupCreate(context, 1, &geom);
      owlGroupBuildAccel(gi);
      owlInstanceGroupSetChild(surfaceModel, meshID, gi);
        
      static size_t sumTriangles = 0;
      sumTriangles += mesh->index.size();
      static size_t sumVertices = 0;
      sumVertices += mesh->vertex.size();
    }

    owlGroupBuildAccel(surfaceModel);

    owlParamsSetGroup(launchParams,"surfaceModel",surfaceModel);

    std::cout << "setting surface model!" << std::endl;
    std::cout << "done setting surface model" << std::endl;
  }
    
  void OptixRenderer::createVolumeBVH()
  {
    OWLVarDecl volumeVars[] = {
      { "frameStateBuffer",         OWL_BUFPTR, OWL_OFFSETOF(VolumeData,frameStateBuffer)},
      { "sameBrickRegionsBuffer",   OWL_BUFPTR, OWL_OFFSETOF(VolumeData,sameBrickRegionsBuffer)},
      { "brickBuffer",              OWL_BUFPTR, OWL_OFFSETOF(VolumeData,brickBuffer)},
      { "valueRangePerBrickBuffer", OWL_BUFPTR, OWL_OFFSETOF(VolumeData,valueRangePerBrickBuffer)},
      { "numChannels",              OWL_INT,    OWL_OFFSETOF(VolumeData,numChannels)},
      { "spaceSkippingEnabled",     OWL_INT,    OWL_OFFSETOF(VolumeData,spaceSkippingEnabled)},
      { /* sentinel to mark end of list */ }
    };

    volumeGeomType = owlGeomTypeCreate(context,
                                       OWL_GEOMETRY_USER,
                                       sizeof(VolumeData),
                                       volumeVars, -1);

    owlGeomTypeSetBoundsProg(volumeGeomType, module,
                             "VolumeBVH");
    owlGeomTypeSetIntersectProg(volumeGeomType, 0, module,
                                "VolumeBVH");
    owlGeomTypeSetClosestHit(volumeGeomType, 0, module,
                             "VolumeBVH");

    volumeGeom = owlGeomCreate(context, volumeGeomType);
#if EXPLICIT_BASIS_METHOD
    owlGeomSetPrimCount(volumeGeom, (int)regions.brickRegions.size());
    owlGeomSetBuffer(volumeGeom,"sameBrickRegionsBuffer",regionsBuffer);
#else
    owlGeomSetPrimCount(volumeGeom, (int)input->bricks.size());
    owlGeomSetBuffer(volumeGeom,"valueRangePerBrickBuffer",valueRangePerBrickBuffer);
#endif
    owlGeomSetBuffer(volumeGeom,"frameStateBuffer",frameStateBuffer);
    owlGeomSetBuffer(volumeGeom,"brickBuffer",brickBuffer);
    owlGeomSet1i(volumeGeom,"numChannels",multiFieldDvr? (int)scalarFields.size(): 1);
    FrameState *fs = (FrameState *)&frameState;
    bool contourPlanesActive = false;
    for (int i=0; i<MAX_CONTOUR_PLANES; ++i) {
      if (fs->contourPlane[i].enabled) {
        contourPlanesActive = true;
        break;
      }
    }
    owlGeomSet1i(volumeGeom,"spaceSkippingEnabled",!contourPlanesActive);
      
    // -------------------------------------------------------
    // create dummy buffers and world so createSurfaces can do launches
    // -------------------------------------------------------
    volumeGroup = owlUserGeomGroupCreate(context, 1, &volumeGeom);

    // compile progs here because we need the bounds prog in accelbuild:
    owlBuildPrograms(context);
    owlGroupBuildAccel(volumeGroup);
    volumeBVH = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(volumeBVH, 0, volumeGroup);
    owlGroupBuildAccel(volumeBVH);

    owlParamsSetGroup(launchParams,"volumeBVH",volumeBVH);
  }
 
        // TODO: dedup
  void OptixRenderer::createIsoSurfaceBVH()
  {
    OWLVarDecl isoVars[] = {
      { "frameStateBuffer",         OWL_BUFPTR, OWL_OFFSETOF(IsoSurfaceData,frameStateBuffer)},
      { "sameBrickRegionsBuffer",   OWL_BUFPTR, OWL_OFFSETOF(IsoSurfaceData,sameBrickRegionsBuffer)},
      { "brickBuffer",              OWL_BUFPTR, OWL_OFFSETOF(IsoSurfaceData,brickBuffer)},
      { "valueRangePerBrickBuffer", OWL_BUFPTR, OWL_OFFSETOF(IsoSurfaceData,valueRangePerBrickBuffer)},
      { /* sentinel to mark end of list */ }
    };

    isoSurfaceGeomType = owlGeomTypeCreate(context,
                                           OWL_GEOMETRY_USER,
                                           sizeof(IsoSurfaceData),
                                           isoVars, -1);

    owlGeomTypeSetBoundsProg(isoSurfaceGeomType, module,
                             "IsoSurface");
    owlGeomTypeSetIntersectProg(isoSurfaceGeomType, 0, module,
                                "IsoSurface");
    owlGeomTypeSetClosestHit(isoSurfaceGeomType, 0, module,
                             "IsoSurface");

    OWLGeom isoSurfaceGeom = owlGeomCreate(context, isoSurfaceGeomType);
#if EXPLICIT_BASIS_METHOD
    owlGeomSetPrimCount(isoSurfaceGeom, (int)regions.brickRegions.size());
    owlGeomSetBuffer(isoSurfaceGeom,"sameBrickRegionsBuffer",regionsBuffer);
#else
    owlGeomSetPrimCount(isoSurfaceGeom, (int)input->bricks.size());
    owlGeomSetBuffer(isoSurfaceGeom,"valueRangePerBrickBuffer",valueRangePerBrickBuffer);
#endif
    owlGeomSetBuffer(isoSurfaceGeom,"frameStateBuffer",frameStateBuffer);
    owlGeomSetBuffer(isoSurfaceGeom,"brickBuffer",brickBuffer);
      
    // -------------------------------------------------------
    // create dummy buffers and world so createSurfaces can do launches
    // -------------------------------------------------------
    isoSurfaceGroup = owlUserGeomGroupCreate(context, 1, &isoSurfaceGeom);

    // compile progs here because we need the bounds prog in accelbuild:
    owlBuildPrograms(context);
    owlGroupBuildAccel(isoSurfaceGroup);
    isoSurfaceBVH = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(isoSurfaceBVH, 0, isoSurfaceGroup);
    owlGroupBuildAccel(isoSurfaceBVH);

    owlParamsSetGroup(launchParams,"isoSurfaceBVH",isoSurfaceBVH);
  }

  void OptixRenderer::createStreamlineBVH()
  {
    OWLVarDecl streamlineVars[] = {
      { "traces",                 OWL_BUFPTR, OWL_OFFSETOF(StreamlineData,traces)},
      { "numTraces",              OWL_INT,    OWL_OFFSETOF(StreamlineData,numTraces)},
      { "numTimesteps",           OWL_INT,    OWL_OFFSETOF(StreamlineData,numTimesteps)},
      { "currentTimestep",        OWL_BUFPTR, OWL_OFFSETOF(StreamlineData,currentTimestep)},
      { /* sentinel to mark end of list */ }
    };

    streamlineGeomType = owlGeomTypeCreate(context,
                                           OWL_GEOMETRY_USER,
                                           sizeof(StreamlineData),
                                           streamlineVars, -1);

    owlGeomTypeSetBoundsProg(streamlineGeomType, module,
                             "Streamline");
    owlGeomTypeSetIntersectProg(streamlineGeomType, 0, module,
                                "Streamline");
    owlGeomTypeSetClosestHit(streamlineGeomType, 0, module,
                             "Streamline");

    streamlineGeom = owlGeomCreate(context, streamlineGeomType);
    owlGeomSetPrimCount(streamlineGeom, traces.numTraces*(traces.numTimesteps-1));
    owlGeomSetBuffer(streamlineGeom,"traces",traces.buffer);
    owlGeomSet1i(streamlineGeom,"numTraces",traces.numTraces);
    owlGeomSet1i(streamlineGeom,"numTimesteps",traces.numTimesteps);
    owlGeomSetBuffer(streamlineGeom,"currentTimestep",traces.currentTimestep);
      
    // -------------------------------------------------------
    // create dummy buffers and world so createSurfaces can do launches
    // -------------------------------------------------------
    streamlineGroup = owlUserGeomGroupCreate(context, 1, &streamlineGeom);

    // compile progs here because we need the bounds prog in accelbuild:
    owlBuildPrograms(context);
    owlGroupBuildAccel(streamlineGroup);
    streamlineBVH = owlInstanceGroupCreate(context,1);
    owlInstanceGroupSetChild(streamlineBVH, 0, streamlineGroup);
    owlGroupBuildAccel(streamlineBVH);

    owlParamsSetGroup(launchParams,"streamlineBVH",streamlineBVH);
  }
} // ::exa
