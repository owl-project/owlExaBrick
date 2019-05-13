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

#include "owl/owl.h"
#include "exa/ExaBricks.h"
#include "exa/KdTree.h"
#include "exa/ScalarField.h"
#include "exa/Texture.h"
#include "exa/TriangleMesh.h"
#if EXPLICIT_BASIS_METHOD
# include "exa/Regions.h"
#endif
#include "programs/FrameState.h"

namespace exa {

  struct OptixRenderer {
    typedef std::shared_ptr<OptixRenderer> SP;

    OptixRenderer(ExaBricks::SP input,
                  std::vector<TriangleMesh::SP> surfaces,
                  std::vector<ScalarField::SP>  scalarField);
   ~OptixRenderer();
    void setVoxelSpaceTransform(const affine3f &voxelSpaceTransform);

    /*! resize the frame buffer */
    void resizeFrameBuffer(void *fbPointer, const owl::vec2i &fbSize);

    void updateIsoValues(const float *isoValues,
                         const int   *channels,
                         const int   *enabled);

    void updateContourPlanes(const vec3f *normals,
                             const float *offsets,
                             const int   *channels,
                             const int   *enabled);

    void updateCamera(const owl::vec3f &pos,
                      const owl::vec3f &dir00,
                      const owl::vec3f &dirDu,
                      const owl::vec3f &dirDv);

    void updateXF(int chan,
                  const float *opacities,
                  const std::vector<owl::vec3f>  &colorMap,
                  const interval<float>     &xfDomain,
                  /*! the value we'll scale the final
                    opacity value of each sample with -
                    allows for making volume more
                    transparent than a 0/1 range for
                    each sample would indicate */
                  float xfOpacityScale=.1f);

    void updateFrameID(int frameID);

    void updateDt(float dt);

    void setSpaceSkipping(bool enable);

    void setGradientShadingDVR(bool enable);

    void setGradientShadingISO(bool enable);

    void setTracerEnabled(bool enable);

    void resetTracer();

    bool advanceTracer();

    void printMemAvail(const std::string &when);

    void render();

    void createSurfaces();
    
    void createVolumeBVH();

    void createIsoSurfaceBVH();

    void createStreamlineBVH();

    
    std::vector<ScalarField::SP> scalarFields;
    
    /*! vector for mapping from scalar field indices to actual scalar
     values - input scalar fields store cells in the order they were
     specified by the simulation, but the bricks contain indices into
     those fields. */
    std::vector<int> indexVector;
    
    ExaBricks::SP    input;
    owl::box3f       voxelSpaceBounds;
    owl::box3f       worldSpaceBounds;

    bool multiFieldDvr = true;

    bool gradientShadingDVR = true;
    bool gradientShadingISO = true;

    /*! Enable / disable space skipping; always disabled in contour
      plane mode */
    bool doSpaceSkipping = true;

    /*! @{ Triggered by transfunc or iso updates; initially false
     as the iso BVH e.g. might not even be available (in the
     non-basis method for example) */
    bool needVolumeBVHRebuild = false;
    bool needIsoBVHRebuild = false;
    bool needStreamlineBVHRebuild = false;
    /*! @} */
    
    std::vector<TriangleMesh::SP> surfaces;
    OWLGeomType surfaceGeomType = 0;
    OWLGroup surfaceGroup = 0;
    OWLGroup surfaceModel = 0;

#if EXPLICIT_BASIS_METHOD
    ExaBrickRegions regions;
    OWLBuffer       regionsBuffer = 0;
    OWLBuffer       regionsLeafListBuffer = 0;
#endif

    struct {
      owl::vec3f origin;
      owl::vec3f normal;
      bool  enabled { false };
      bool  dirty   {  true };
    } clipPlane;

    exa::FrameState frameState;
    owl::vec2i fbSize;
    exa::Texture transFunc[MAX_CHANNELS];

    OWLContext             context = 0;
    OWLModule              module = 0;
    OWLBuffer              colorBuffer = 0;
    OWLBuffer              accumBuffer = 0;
    OWLBuffer              frameStateBuffer = 0;

    struct {
      int tracerEnabled = false;
      vec3i tracerChannels { 0,1,2 };
      int numTraces = 1000;
      int numTimesteps = 100;
      float steplen = 1e-6f;
      OWLBuffer buffer;
      OWLBuffer currentTimestep;
      int timestepHost=0;
      box3f seedRegion {{.3f,.3f,.5f},{.8f,.8f,.5f}};
    } traces;

    OWLBuffer              scalarBuffers = 0;
    OWLBuffer              scalarBufferOffsets = 0;

    OWLBuffer              primaryChannelsBuffer = 0;
    
    OWLBuffer              brickBuffer = 0;
    // That's only used in nearest interpolation mode
    OWLBuffer              valueRangePerBrickBuffer = 0;
    OWLRayGen              raygenProgram = 0;
    OWLLaunchParams        launchParams = 0;
    OWLGeomType            volumeGeomType = 0;
    OWLGeom                volumeGeom = 0;
    OWLGroup               volumeGroup = 0;
    OWLGroup               volumeBVH = 0;
    OWLGeomType            isoSurfaceGeomType = 0;
    OWLGroup               isoSurfaceGroup = 0;
    OWLGroup               isoSurfaceBVH = 0;
    OWLGeomType            streamlineGeomType = 0;
    OWLGeom                streamlineGeom = 0;
    OWLGroup               streamlineGroup = 0;
    OWLGroup               streamlineBVH = 0;
  };
  
} // ::exa
