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

#include "owl/common/math/random.h"
#include "owl/owl.h"
#include "Brick.h"
#include "Camera.h"
#include "FrameState.h"
#include "LaunchParams.h"
#include "IsoSurfaceData.h"
#include "StreamlineData.h"
#include "SurfaceGeomData.h"
#include "VolumeData.h"
#include "../exa/Regions.h"

#define ISO_SURFACES 1

/*! primitive ID code we use to indicate a stream line intersection */
#define PRIMID_STREAMLINE -25

/*! primitive ID code we use to indicate a plane intersection */
#define PRIMID_PLANE      -24

/*! primitive ID code we use to indicate a iso-surface intersection */
#define PRIMID_ISOSURFACE -23

namespace exa {
  /*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  typedef owl::LCG<16> Random;

  // Early ray termination threshold
#define TERMINATION_THRESHOLD 0.98f

#define PROPER_DISTANCE_IN_OPACITY_CORRECTION 1
  
  inline __device__
  float linear_to_srgb(float x)
  {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
  }

  inline __device__
  int32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __device__
  int32_t make_rgba8(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xff << 24);
  }

  inline __device__
  void make_orthonormal_basis(vec3f& u, vec3f& v, const vec3f& w)
  {
    v = abs(w.x) > abs(w.y)?normalize(vec3f(-w.z,0,w.x)):normalize(vec3f(0,w.z,-w.y));
    u = cross(v, w);
  }

  inline __device__
  vec3f cosine_sample_hemisphere(float u1, float u2)
  {
    float r     = sqrtf(u1);
    float theta = 2.f*M_PI*u2;
    float x     = r * cosf(theta);
    float y     = r * sinf(theta);
    float z     = sqrtf(1.f-u1);
    return vec3f(x,y,z);
  }
  
  inline __device__
  float saturate(float x)
  {
    return min(1.f,(max(0.f,x)));
  }

  //------------------------------------------------------------------------------------------
  // Hue and temperature to RGB
  //------------------------------------------------------------------------------------------

  inline __device__
  vec3f hue_to_rgb(float hue)
  {
    using T = float;

    T s = saturate( hue ) * T(6.0f);

    T r = saturate( abs(s - T(3.0)) - T(1.0) );
    T g = saturate( T(2.0) - fabsf(s - T(2.0)) );
    T b = saturate( T(2.0) - fabsf(s - T(4.0)) );

    return vec3f(r, g, b);
  }

  inline __device__
  vec3f temperature_to_rgb(float t)
  {
    using T = float;

    T K = T(4.0f / 6.0f);

    T h = K - K * t;
    T v = T(0.5f) + T(0.5f) * t;

    return v * hue_to_rgb(h);
  }

  /*! lookup, clampoing, and inteprolation by
    texture hardware .... */
  inline __device__
  vec4f lookupTransferFunction(const FrameState &fs,
                               float in_scalar,
                               int channel)
  {
    float scalar
      = (NUM_XF_VALUES-1)
      * (in_scalar - fs.xfDomain[channel].lower)
      / ((fs.xfDomain[channel].upper - fs.xfDomain[channel].lower)+1e-20f);
    scalar = clamp(scalar+.5f, 0.f, NUM_XF_VALUES - 1.f);
    scalar /=  NUM_XF_VALUES - 1.f;
    
    vec4f xfResult = tex1D<float4>(fs.xfTexture[channel],scalar);
    xfResult.w *= fs.xfOpacityScale;
    return xfResult;
  }

  /*! create a new ray for current pixel, using RNG to generate
    intra-pixel offset ofr super-sampling */
  inline __device__
  owl::Ray generateRay(Random &rnd)
  {
    const FrameState &fs = optixLaunchParams.frameStateBuffer[0];
    const vec2i pixelID = owl::getLaunchIndex();
    vec3f org = fs.camera.pos;
    vec3f dir
      = fs.camera.dir00
      + (pixelID.x+rnd())*fs.camera.dirDu
      + (pixelID.y+rnd())*fs.camera.dirDv
      ;

    return owl::Ray((float3)org,
                    (float3)normalize(dir),
                    /* tmin     : */ 0.f,
                    /* tmax     : */ 2e10f);
  }

  // ==================================================================
  // intersect and bounds program to build and intersect the BVH -
  // either brick or region bvh
  // ==================================================================

  /*! per ray data for a volume query ray - returns ID of leaf, and
    [t0,t1] interval that ray overlaps this leaf */
  struct VolumePRD {
    int leafID;
    float t0, t1;
  };

  inline __device__ VolumePRD traceVolumeRay(const owl::Ray &ray)
  {
    VolumePRD prd;
    prd.leafID = -1;
    prd.t0 = prd.t1 = 0.f; // doesn't matter as long as leafID==-1
    owl::traceRay(optixLaunchParams.volumeBVH, ray, prd,
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT);
    return prd;
  }

  /*! compute ray-box test, and return true if ray intersects this
    box. if ray does intersect box, return [t0,t1] interval of
    overlap in refernce paramters */
  inline __device__
  bool boxTest(const owl::Ray &ray, const box3f &box,
               float &t0, float &t1)
  {
    const vec3f t_lo = (box.lower - ray.origin) / ray.direction;
    const vec3f t_hi = (box.upper - ray.origin) / ray.direction;

    const vec3f t_nr = min(t_lo,t_hi);
    const vec3f t_fr = max(t_lo,t_hi);

    t0 = max(ray.tmin,reduce_max(t_nr));
    t1 = min(ray.tmax,reduce_min(t_fr));
    return t0 < t1;
  }

  /*! intersection program for volume queries */
  OPTIX_INTERSECT_PROGRAM(VolumeBVH)()
  {
    const VolumeData &self = owl::getProgramData<VolumeData>();
    int   leafID = optixGetPrimitiveIndex();
    owl::Ray ray(optixGetWorldRayOrigin(),
                 optixGetWorldRayDirection(),
                 optixGetRayTmin(),
                 optixGetRayTmax());
    float t0 = ray.tmin, t1 = ray.tmax;
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = self.sameBrickRegionsBuffer[leafID];
    const box3f             bounds = region.domain;
#else
    const Brick &brick = self.brickBuffer[leafID];
    const box3f bounds = brick.getBounds();
#endif
    if (!boxTest(ray,bounds,t0,t1))
      return;

    if (optixReportIntersection(t0, 0)) {
      VolumePRD& prd = owl::getPRD<VolumePRD>();
      prd.t0 = t0;
      prd.t1 = t1;
      prd.leafID = leafID;
    }
  }


  inline __device__
  float clampf(const float f, const float lo, const float hi)
  { return min(hi,max(lo,f)); }
    
  inline __device__
  int clampi(const int f, const int lo, const int hi)
  { return min(hi,max(lo,f)); }


  inline __device__
  bool activeForVolumeSampling(const VolumeData &self,
                               const range1f &valueRange,
                               int channel)
  {
    const auto &global = self.frameStateBuffer[0];
    
    if (valueRange.lower > global.xfDomain[channel].upper)
      return false;
    if (valueRange.upper < global.xfDomain[channel].lower)
      return false;

    const float scaled_lo
      = (valueRange.lower - global.xfDomain[channel].lower)
      / ((global.xfDomain[channel].upper - global.xfDomain[channel].lower)+1e-20f);
    const float scaled_hi
      = (valueRange.upper - global.xfDomain[channel].lower)
      / ((global.xfDomain[channel].upper - global.xfDomain[channel].lower)+1e-20f);

    const int idx_lo = clampi(int(scaled_lo*(NUM_XF_VALUES-1)),0,NUM_XF_VALUES-1);
    const int idx_hi = clampi(int(scaled_hi*(NUM_XF_VALUES-1))+1,0,NUM_XF_VALUES-1);
    for (int i=idx_lo;i<=idx_hi;i++) {
      float cellValue = float(i)/(NUM_XF_VALUES-1);
      cellValue *= global.xfDomain[channel].upper - global.xfDomain[channel].lower;
      cellValue += global.xfDomain[channel].lower;
      vec4f rgba = lookupTransferFunction(self.frameStateBuffer[0],cellValue,channel);
      if (rgba.w > 0.f)
        return true;
    }
      
    return false;
  }

  /*! bounds program for volume bricks/domains, to allow optix to
    build a BVH over those bricks */
  OPTIX_BOUNDS_PROGRAM(VolumeBVH)(const void* geomData,
                                  box3f& result,
                                  int leafID)
  {
    const VolumeData &self = *(const VolumeData*)geomData;
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = self.sameBrickRegionsBuffer[leafID];
    if (leafID == 0) {
      if (self.spaceSkippingEnabled)
        printf("rebuilding volume bvh, space skipping is enabled...\n");
      else
        printf("rebuilding volume bvh, space skipping is disabled...\n");
    }

    bool active = false;
    for (int c=0; c<self.numChannels; ++c) {
      active |= activeForVolumeSampling(self, region.valueRange, c);
      if (active) break;
    }
   
    if (self.spaceSkippingEnabled) {
      if (active)
        result = region.domain;
      else
        result = box3f(vec3f(+1e-6f),vec3f(-1e-6f));
    } else {
      result = region.domain;
    }
#else
    if (self.spaceSkippingEnabled) {
      bool active = false;
      for (int c=0; c<self.numChannels; ++c) {
        active |= activeForVolumeSampling(self, region.valueRangePerBrickBuffer[leafID], c);
        if (active) break;
      }
      const bool active = activeForVolumeSampling(self, self.valueRangePerBrickBuffer[leafID]);
      if (active) {
        const Brick &brick = self.brickBuffer[leafID];
        result = brick.getBounds();
      } else
        result = box3f(vec3f(+1e-6f),vec3f(-1e-6f));
    } else {
      const Brick &brick = self.brickBuffer[leafID];
      result = brick.getBounds();
    }
#endif
  }

  OPTIX_CLOSEST_HIT_PROGRAM(VolumeBVH)()
  { /* empty */ }


  // ==================================================================
  // closest hit program to find closest *iso*-surface intsection
  // ==================================================================
  
  struct IsoSurfacePRD {
    int leafID;
    float t0, t1;
  };
  
  OPTIX_INTERSECT_PROGRAM(IsoSurface)()
  {
    const IsoSurfaceData& self = owl::getProgramData<IsoSurfaceData>();
    int leafID = optixGetPrimitiveIndex();
    owl::Ray ray(optixGetWorldRayOrigin(),
                 optixGetWorldRayDirection(),
                 optixGetRayTmin(),
                 optixGetRayTmax());
    float t0 = ray.tmin, t1 = ray.tmax;
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = self.sameBrickRegionsBuffer[leafID];
    const box3f             bounds = region.domain;
#else
    const Brick &brick = self.brickBuffer[leafID];
    const box3f bounds = brick.getBounds();
#endif
    if (!boxTest(ray,bounds,t0,t1))
      return;

    if (optixReportIntersection(t0, 0)) {
      IsoSurfacePRD& prd = owl::getPRD<IsoSurfacePRD>();
      prd.t0 = t0;
      prd.t1 = t1;
      prd.leafID = leafID;
    }
  }

  OPTIX_BOUNDS_PROGRAM(IsoSurface)(const void* geomData,
                                   box3f& result,
                                   int leafID)
  {
    if (leafID == 0)
      printf("rebuilding isosurface bvh...\n");

    const IsoSurfaceData &self = *(IsoSurfaceData*)geomData;
    const auto &global = self.frameStateBuffer[0];
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = self.sameBrickRegionsBuffer[leafID];
    const box3f bounds = region.domain;
    const range1f range = region.valueRange;
#else
    const Brick &brick = self.brickBuffer[leafID];
    const box3f bounds = brick.getBounds();
    const range1f range = self.valueRangePerBrickBuffer[leafID];
#endif
    bool active = false;
    for (int i=0;i<MAX_ISO_SURFACES;i++) {
      if (global.isoSurface[i].enabled &&
          global.isoSurface[i].value >= range.lower &&
          global.isoSurface[i].value <= range.upper)
        active = true;
    }
    result
      = active
      ? bounds
      : box3f();
  }

  OPTIX_CLOSEST_HIT_PROGRAM(IsoSurface)()
  { /* empty */ }


  // ==================================================================
  // closest hit program to find closest surface intsection
  // ==================================================================

  struct SurfacePRD {
    int primID;
    float t_hit;
    vec3f Ng;
    float ambient;
    vec3f baseColor;
  };
 
  OPTIX_CLOSEST_HIT_PROGRAM(SurfaceBVH)()
  {
    const SurfaceGeomData& self = owl::getProgramData<SurfaceGeomData>();
    SurfacePRD& surfacePRD = owl::getPRD<SurfacePRD>();
    surfacePRD.t_hit = optixGetRayTmax();
    surfacePRD.primID = optixGetPrimitiveIndex();
    vec3i index = self.indexBuffer[surfacePRD.primID];
    vec3f A = self.vertexBuffer[index.x];
    vec3f B = self.vertexBuffer[index.y];
    vec3f C = self.vertexBuffer[index.z];
    surfacePRD.Ng = normalize(cross(B-A,C-A));
    surfacePRD.ambient = .2f;
    surfacePRD.baseColor = vec3f(.8f);
  }


  // ==================================================================
  // Helpers and OptiX programs for streamline rendering
  // ==================================================================
  
  /* ray - rounded cone intersection. */
  inline __device__
  bool intersectRoundedCone(
                            const vec3f  pa, const vec3f  pb,
                            const float  ra, const float  rb,
                            const owl::Ray ray,
                            float& hit_t,
                            vec3f& isec_normal)
  {
    /* iw, mar 1st, 2020 - this is the version that uses the
       numerical accuracy trick of firs tmoving the ray origin
       closer to the object, then computing distance (which is
       numerically unstable for large distances), and then moving
       the origin back _after_ the dsitance has been found. this
       fixes some rpetty egregious numerical issues we've seen, BUT
       i'm not 100% sure what happens in cases where we're close to
       the object, etc (it migth be that the "if < 0" etc tests will
       then reject a hitpointe _before_ we can move the origin back
       .... i don't _think_ that happens, but .... huh. */
    vec3f ro = ray.origin;
    const vec3f &rd = ray.direction;
    float minDist = max(0.f,min(length(pa-ro)-ra,length(pb-ro)-rb));
    ro = ro + minDist *rd;

    vec3f  ba = pb - pa;
    vec3f  oa = ro - pa;
    vec3f  ob = ro - pb;
    float  rr = ra - rb;
    float  m0 = dot(ba, ba);
    float  m1 = dot(ba, oa);
    float  m2 = dot(ba, rd);
    float  m3 = dot(rd, oa);
    float  m5 = dot(oa, oa);
    float  m6 = dot(ob, rd);
    float  m7 = dot(ob, ob);

    float d2 = m0 - rr * rr;

    float k2 = d2 - m2 * m2;
    float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0f - m0 * ra * ra;

    float h = k1 * k1 - k0 * k2;
    if (h < 0.0f) return false;
    float t = (-sqrtf(h) - k1) / k2;

    float y = m1 - ra * rr + t * m2;
    if (y > 0.0f && y < d2) {
      hit_t = minDist + t;
      isec_normal = (d2 * (oa + t * rd) - ba * y);
      return true;
    }

    // Caps. 
    float h1 = m3 * m3 - m5 + ra * ra;
    if (h1 > 0.0f) {
      t = -m3 - sqrtf(h1);
      hit_t = minDist + t;
      isec_normal = ((oa + t * rd) / ra);
      return true;
    }
    return false;
  }

  OPTIX_INTERSECT_PROGRAM(Streamline)()
  {
    int primID = optixGetPrimitiveIndex();
    const auto& self
      = owl::getProgramData<StreamlineData>();

    owl::Ray ray(optixGetWorldRayOrigin(),
                 optixGetWorldRayDirection(),
                 optixGetRayTmin(),
                 optixGetRayTmax());
    
    SurfacePRD &prd = owl::getPRD<SurfacePRD>();
    
    float tmp_hit_t = ray.tmax;

    const vec3f pa = self.traces[primID];
    const vec3f pb = self.traces[primID+1];
    const float ra = 2.f;
    const float rb = 2.f;
    
    vec3f normal;
    
    if (intersectRoundedCone(pa, pb, ra, rb, ray, tmp_hit_t, normal)) {
      if (optixReportIntersection(tmp_hit_t, 0)) {
        prd.primID = PRIMID_STREAMLINE;
        prd.t_hit = tmp_hit_t;
        prd.Ng = normalize(normal);
        prd.baseColor = vec3f(.8f);
      }
    }
  }

  OPTIX_BOUNDS_PROGRAM(Streamline)(const void* geomData,
                                   box3f& primBounds,
                                   int primID)
  {
    const StreamlineData &self = *(const StreamlineData*)geomData;

    const int timestepID = primID % self.numTimesteps;

    const int t = self.currentTimestep[0];

    if (timestepID >= t-1) {
      primBounds = box3f(vec3f(+1e-6f),vec3f(-1e-6f));
      return;
    }
      
    vec3f pa = self.traces[primID];
    vec3f ra = 2.f;

    vec3f pb = self.traces[primID+1];
    vec3f rb = 2.f;

    if (pa.x < 2e10f && pb.x < 2e10f) {
      primBounds
        = box3f()
        .including(pa-ra)
        .including(pa+ra);
      
      primBounds
        = primBounds
        .including(pb-rb)
        .including(pb+rb);
    } else {
      primBounds = box3f(vec3f(+1e-6f),vec3f(-1e-6f));
    }
  }

  OPTIX_CLOSEST_HIT_PROGRAM(Streamline)()
  { /* empty */ }


  // ==================================================================
  // actual rendering code
  // ==================================================================
  
#if EXPLICIT_BASIS_METHOD
  inline __device__ float getScalar(const int brickID,
                                    const int ix, const int iy, const int iz,
                                    const int channel)
  {
    unsigned offset = optixLaunchParams.scalarBufferOffsets[channel];

    const Brick &brick = optixLaunchParams.brickBuffer[brickID];
    const int idx
      = brick.begin
      + ix
      + iy * brick.size.x
      + iz * brick.size.x*brick.size.y;
    return optixLaunchParams.scalarBuffers[offset + idx];
  }

  inline void __both__ add(float &sumWeights,
                           float &sumWeightedValues,
                           float weight,
                           float scalar)
  {
    sumWeights += weight;
    sumWeightedValues += weight*scalar;
  }

  inline void __both__ add(vec3f &sumWeights,
                           vec3f &sumWeightedValues,
                           vec3f weight,
                           vec3f scalar)
  {
    sumWeights += weight;
    sumWeightedValues += weight*scalar;
  }

#if ALLOW_EMPTY_CELLS
  inline __device__ bool notEmptyCell(float value) { return value != EMPTY_CELL_POISON_VALUE; }
#else
  inline __device__ bool notEmptyCell(float ) { return true; }
#endif
  
  template<bool NEED_DERIVATIVE=true>
  inline __device__ void addBasisFunctions(float &sumWeightedValues,
                                           float &sumWeights,
                                           vec3f &sumDerivatives,
                                           vec3f &sumDerivativeCoefficients,
                                           const int brickID,
                                           const vec3f pos,
                                           const int channel)
  {
    const Brick &brick    = optixLaunchParams.brickBuffer[brickID];
    const float cellWidth = (1<<brick.level);

    const vec3f localPos = (pos - vec3f(brick.lower)) / vec3f(cellWidth) - vec3f(0.5f);
    vec3i idx_lo   = vec3i(floorf(localPos.x),floorf(localPos.y),floorf(localPos.z));
    idx_lo = max(vec3i(-1), idx_lo);
    const vec3i idx_hi   = idx_lo + vec3i(1);

    const vec3f frac     = localPos - vec3f(idx_lo);
    const vec3f neg_frac = vec3f(1.f) - frac;

    // #define INV_CELL_WIDTH invCellWidth
#define INV_CELL_WIDTH 1.f
    if (idx_lo.z >= 0 && idx_lo.z < brick.size.z) {
      if (idx_lo.y >= 0 && idx_lo.y < brick.size.y) {
        if (idx_lo.x >= 0 && idx_lo.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_lo.x,idx_lo.y,idx_lo.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (neg_frac.z)*(neg_frac.y)*(neg_frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_lo.y,idx_lo.z);
            if (NEED_DERIVATIVE) {
              const float dx = (neg_frac.z)*(neg_frac.y)*(-INV_CELL_WIDTH);
              const float dy = (neg_frac.z)*(neg_frac.x)*(-INV_CELL_WIDTH);
              const float dz = (neg_frac.y)*(neg_frac.x)*(-INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
        if (idx_hi.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_hi.x,idx_lo.y,idx_lo.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (neg_frac.z)*(neg_frac.y)*(frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_lo.y,idx_lo.z);
            if (NEED_DERIVATIVE) {
              const float dx = (neg_frac.z)*(neg_frac.y)*(+INV_CELL_WIDTH);
              const float dy = (neg_frac.z)*(    frac.x)*(-INV_CELL_WIDTH);
              const float dz = (neg_frac.y)*(    frac.x)*(-INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
      }
      if (idx_hi.y < brick.size.y) {
        if (idx_lo.x >= 0 && idx_lo.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_lo.x,idx_hi.y,idx_lo.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (neg_frac.z)*(frac.y)*(neg_frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_hi.y,idx_lo.z);
            if (NEED_DERIVATIVE) {
              const float dx = (neg_frac.z)*(    frac.y)*(-INV_CELL_WIDTH);
              const float dy = (neg_frac.z)*(neg_frac.x)*(+INV_CELL_WIDTH);
              const float dz = (    frac.y)*(neg_frac.x)*(-INV_CELL_WIDTH);
              const float scalar = getScalar(brickID,idx_lo.x,idx_hi.y,idx_lo.z, channel);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
        if (idx_hi.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_hi.x,idx_hi.y,idx_lo.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (neg_frac.z)*(frac.y)*(frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_hi.y,idx_lo.z);
            if (NEED_DERIVATIVE) {
              const float dx = (neg_frac.z)*(    frac.y)*(+INV_CELL_WIDTH);
              const float dy = (neg_frac.z)*(    frac.x)*(+INV_CELL_WIDTH);
              const float dz = (    frac.y)*(    frac.x)*(-INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
      }
    }
    
    if (idx_hi.z < brick.size.z) {
      if (idx_lo.y >= 0 && idx_lo.y < brick.size.y) {
        if (idx_lo.x >= 0 && idx_lo.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_lo.x,idx_lo.y,idx_hi.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (frac.z)*(neg_frac.y)*(neg_frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_lo.y,idx_hi.z);
            if (NEED_DERIVATIVE) {
              const float dx = (    frac.z)*(neg_frac.y)*(-INV_CELL_WIDTH);
              const float dy = (    frac.z)*(neg_frac.x)*(-INV_CELL_WIDTH);
              const float dz = (neg_frac.y)*(neg_frac.x)*(+INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
        if (idx_hi.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_hi.x,idx_lo.y,idx_hi.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (frac.z)*(neg_frac.y)*(frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_lo.y,idx_hi.z);
            if (NEED_DERIVATIVE) {
              const float dx = (    frac.z)*(neg_frac.y)*(+INV_CELL_WIDTH);
              const float dy = (    frac.z)*(    frac.x)*(-INV_CELL_WIDTH);
              const float dz = (neg_frac.y)*(    frac.x)*(+INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
      }
      if (idx_hi.y < brick.size.y) {
        if (idx_lo.x >= 0 && idx_lo.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_lo.x,idx_hi.y,idx_hi.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (frac.z)*(frac.y)*(neg_frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_lo.x,idx_hi.y,idx_hi.z);
            if (NEED_DERIVATIVE) {
              const float dx = (    frac.z)*(    frac.y)*(-INV_CELL_WIDTH);
              const float dy = (    frac.z)*(neg_frac.x)*(+INV_CELL_WIDTH);
              const float dz = (    frac.y)*(neg_frac.x)*(+INV_CELL_WIDTH);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
        if (idx_hi.x < brick.size.x) {
          const float scalar = getScalar(brickID,idx_hi.x,idx_hi.y,idx_hi.z, channel);
          if (notEmptyCell(scalar)) {
            const float weight = (frac.z)*(frac.y)*(frac.x);
            // sumWeights += weight;
            // sumWeightedValues += weight * getScalar(brickID,idx_hi.x,idx_hi.y,idx_hi.z);
            if (NEED_DERIVATIVE) {
              const float dx = (    frac.z)*(    frac.y)*(+INV_CELL_WIDTH);
              const float dy = (    frac.z)*(    frac.x)*(+INV_CELL_WIDTH);
              const float dz = (    frac.y)*(    frac.x)*(+INV_CELL_WIDTH);
              const float scalar = getScalar(brickID,idx_hi.x,idx_hi.y,idx_hi.z, channel);
              add(sumDerivativeCoefficients,sumDerivatives,vec3f(dx,dy,dz),scalar);
            }
            add(sumWeights,sumWeightedValues,weight,scalar);
          }
        }
      }
    }
  }
#endif

  // Sample point
  inline __device__ bool samplePoint(float &value,
                                     const int leafID,
                                     const vec3f pos,
                                     const int channel)
  {
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = optixLaunchParams.sameBrickRegionsBuffer[leafID];
    const int *childList  = &optixLaunchParams.sameBrickRegionsLeafList[region.leafListBegin];
    const int  childCount = region.leafListSize;
    float sumWeightedValues = 0.f;
    float sumWeights = 0.f;
    vec3f sumDerivatives(0.f);
    vec3f sumDerivativeCoefficients(0.f);
    for (int childID=0;childID<childCount;childID++) {
      const int    brickID = childList[childID];
      addBasisFunctions<false>(sumWeightedValues, sumWeights, sumDerivatives, sumDerivativeCoefficients,
                               brickID, pos, channel);
    }

    if (sumWeights <= 1e-20f) {
      // Not a valid sample if sum weights near 0
      return false;
    } else {
      value = sumWeightedValues / sumWeights;
      return true;
    }
#else
    const Brick &brick = optixLaunchParams.brickBuffer[leafID];
    const vec3i localIdx
      = max(vec3i(0),min(brick.size-1,(vec3i(pos) - brick.lower) / (1<<brick.level)));
    unsigned offset = optixLaunchParams.scalarBufferOffsets[channel];
    const int cellIndex = brick.getIndexIndex(localIdx);
    value = optixLaunchParams.scalarBuffers[offset + cellIndex];
    return true;
#endif
  }

  inline __device__ float samplePointWithInfRay(const vec3f pos, int channel)
  {
    owl::Ray ray;
    ray.origin = pos;
    ray.direction = vec3f(1.f);
    ray.tmin = 0.f;
    ray.tmax = 2e-10f;
    VolumePRD prd = traceVolumeRay(ray);

    float value;
    samplePoint(value,prd.leafID,pos,channel);
    return value;
  }

  // Compute gradient with central differences
  template<bool ISO>
  inline __device__
  vec3f gradientCD(const vec3f pos,
                   const int posLeafID,
                   const float delta,
                   const int channel)
  {
    const vec3f dt[]
      = {
         { delta, 0.f, 0.f },
         { 0.f, delta, 0.f },
         { 0.f, 0.f, delta },
    };

    vec3f s[2];
    for (int i=0; i<3; ++i) {
      for (int j=0; j<2; ++j) {
#if FAST_CENTRAL_DIFFERENCES
        const vec3f samplePos = j==0? pos+dt[i]: pos-dt[i];
        samplePoint(s[j][i],posLeafID,samplePos,channel);
#else
        owl::Ray ray;
        ray.origin = j==0? pos+dt[i]: pos-dt[i];
        ray.direction = vec3f(1.f);
        ray.tmin = 0.f;
        ray.tmax = 2e-10f;
        int leafID = -1;
        if (ISO) {
          IsoSurfacePRD prd;
          prd.leafID = -1;
          prd.t0 = prd.t1 = 0.f; // doesn't matter as long as leafID==-1
          owl::traceRay(optixLaunchParams.isoSurfaceBVH, ray, prd,
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT);
          leafID = prd.leafID;
        } else {
          VolumePRD prd = traceVolumeRay(ray);
          leafID = prd.leafID;
        }

        if (leafID >= 0)
          samplePoint(s[j][i],leafID,ray.origin,channel);
        else
          s[j][i] = 0.f;
#endif
      }
    }
    return s[1] - s[0];
  }

  //! Sample point, and compute derivative
  template<bool ISO>
  inline __device__
  bool samplePointWithDerivative(float &value,
                                 vec3f &derivatives,
                                 const int leafID,
                                 const vec3f pos,
                                 const int channel)
  {
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = optixLaunchParams.sameBrickRegionsBuffer[leafID];
    const int *childList  = &optixLaunchParams.sameBrickRegionsLeafList[region.leafListBegin];
    const int  childCount = region.leafListSize;
    float sumWeightedValues = 0.f;
    float sumWeights = 0.f;
    vec3f sumDerivatives(0.f);
    vec3f sumDerivativeCoefficients(0.f);
    for (int childID=0;childID<childCount;childID++) {
      const int    brickID = childList[childID];
#if ANALYTIC_GRADIENTS
      addBasisFunctions(sumWeightedValues, sumWeights, sumDerivatives, sumDerivativeCoefficients,
                        brickID, pos, channel);
#else
      addBasisFunctions<false>(sumWeightedValues, sumWeights, sumDerivatives, sumDerivativeCoefficients,
                               brickID, pos, channel);
#endif
    }

    if (sumWeights <= 1e-20f) {
      // Not a valid sample if sum weights near 0
      return false;
    } else {
      value = sumWeightedValues / sumWeights;
#if ANALYTIC_GRADIENTS
      derivatives = vec3f(sumWeights*sumDerivatives.x-sumWeightedValues*sumDerivativeCoefficients.x,
                          sumWeights*sumDerivatives.y-sumWeightedValues*sumDerivativeCoefficients.y,
                          sumWeights*sumDerivatives.z-sumWeightedValues*sumDerivativeCoefficients.z);
      // Drop the _denominator_ of the quotient rule as this
      // only influences gradient magnitude and not direction:
      //            derivatives /= (sumWeights*sumWeights+1e-20f);
#else
      const float delta = (region.finestLevelCellWidth+1)*.5f;
      // Compute derivatives using point samples
      derivatives = gradientCD<ISO>(pos,leafID,delta,channel);
#endif
      return true;
    }
#else
    const Brick &brick = optixLaunchParams.brickBuffer[leafID];
    const vec3i localIdx
      = max(vec3i(0),min(brick.size-1,(vec3i(pos) - brick.lower) / (1<<brick.level)));
    unsigned offset = optixLaunchParams.scalarBufferOffsets[channel];
    const int cellIndex = brick.getIndexIndex(localIdx);
    value = optixLaunchParams.scalarBuffers[offset + cellIndex];

    const float delta = (brick.level+1)*.5f;
    // Compute derivatives using point samples
    derivatives = gradientCD<ISO>(pos,leafID,delta,channel);
    return true;
#endif
  }


  inline __device__
  bool sampleDirection(const vec3f pos,
                       vec3f &result)
  {
    owl::Ray ray;
    ray.origin = pos;
    ray.direction = vec3f(1.f);
    ray.tmin = 0.f;
    ray.tmax = 2e-10f;
    VolumePRD prd = traceVolumeRay(ray);

    for (int i=0; i<3; ++i) {
      int channel = optixLaunchParams.tracerChannels[i];
      if (!samplePoint(result[i],prd.leafID,pos,channel))
        return false;
    }

    return true;
  }


  // Result variables from integration
  struct IntegrationResult {

    inline __device__
    IntegrationResult(vec4f &pc,
                      float thit=-1.f,
                      vec3f grad=vec3f(0.f))
      : pixelColor(pc)
      , t_hit(thit)
      , gradient(grad)
    {}

    // Pixel color, this field is *updated*
    vec4f &pixelColor;

    // Hit position, -1 if not applicable
    float t_hit;

    // Gradient, zero-length if not applicable
    vec3f gradient;
  };

  inline __device__
  void integrateVolume(const owl::Ray &ray,
                       IntegrationResult &result,
                       float t_sample,
                       float actual_dt,
                       float cellValue,
                       const vec3f &gradient,
                       int finestLevelCellWidth,
                       int leafID,
                       int channel)
  {
    if (actual_dt == 0.f)
      return;

    vec4f &pixelColor = result.pixelColor;
    vec4f sample = lookupTransferFunction(optixLaunchParams.frameStateBuffer[0], cellValue, channel);
    if (length(gradient) > finestLevelCellWidth*1e-6f) {
      const vec3f lightDir = -vec3f(ray.direction);
      const float scale
        = fabsf(dot(lightDir,gradient))
        / sqrtf(dot(gradient,gradient)*dot(lightDir,lightDir));
      (vec3f&)sample *= scale;
    }
    sample.w    = 1.f - powf(1.f-sample.w, actual_dt);
    pixelColor += (1.f-pixelColor.w)*sample.w*vec4f(vec3f(sample), 1.f);
    // Will note: Engel Real time volume graphics also mentions applying a scaling
    // to the sampled color, but this seems to make it worse actually (darkens the artifacts)
    //pixelColor += (1.f-pixelColor.w)*sample.w*vec4f(vec3f(sample) * actual_dt / base_dt),1.f);
  }

  struct IsoSurfaceIntegrationFunction {
    inline __device__ IsoSurfaceIntegrationFunction()
      : lastCellValue(-1e36f)
      , lastGradient(.0f)
    {}

    inline __device__
    void operator()(const owl::Ray &ray,
                    IntegrationResult &result,
                    float t_sample,
                    float actual_dt,
                    float cellValue,
                    const vec3f &gradient,
                    int finestLevelCellWidth,
                    int leafID,
                    int channel)
    {
      if (lastCellValue >= -1e35f) {
        const FrameState &fs = optixLaunchParams.frameStateBuffer[0];
        for (int i=0;i<MAX_ISO_SURFACES;i++) {
          if (optixLaunchParams.frameStateBuffer[0].isoSurface[i].enabled &&
              optixLaunchParams.frameStateBuffer[0].isoSurface[i].channel == channel &&
              ((lastCellValue<=fs.isoSurface[i].value && cellValue>=fs.isoSurface[i].value)
               || (lastCellValue>=fs.isoSurface[i].value && cellValue<=fs.isoSurface[i].value)))
            {
              // Weighted average between this and the last
              // sample, but put more emphasize on the one
              // that is closer (wrt. euclidian distance) to
              // isovalue
              float iso = fs.isoSurface[i].value;
              float d1=fabsf(lastCellValue-iso);
              float d2=fabsf(cellValue-iso);
              float w1=1.f-d1/(d1+d2);
              float w2=1.f-d2/(d1+d2);

              float tavg = last_t*w1+t_sample*w2;
              float cellValue = 0.f;
              vec3f grad = vec3f(0.f);
              vec4f sample = vec4f(1.f, 0.f, 0.f, 1.f);

              const vec3f isopt = ray.origin + tavg * ray.direction;

              if (optixLaunchParams.gradientShadingISO) {
                if (samplePointWithDerivative<true>(cellValue, grad, leafID, isopt,
                                                    fs.isoSurface[i].channel)) {
                  sample = lookupTransferFunction(optixLaunchParams.frameStateBuffer[0],
                                                  cellValue, fs.isoSurface[i].channel);
                  grad = normalize(grad);

                  // Make the normal face forward
                  if (dot(grad, vec3f(ray.direction)) > 0.f) {
                    grad = -grad;
                  }
                }
              } else {
                if (samplePoint(cellValue, leafID, isopt, fs.isoSurface[i].channel)) {
                  sample = lookupTransferFunction(optixLaunchParams.frameStateBuffer[0],
                                                  cellValue, fs.isoSurface[i].channel);
                }
              }

              if (optixLaunchParams.colormapChannel != 0) {
                cellValue = 0.f;
                if (samplePoint(cellValue, leafID, isopt, optixLaunchParams.colormapChannel)) {
                  sample = lookupTransferFunction(optixLaunchParams.frameStateBuffer[0],
                                                  cellValue, 0);
                }
              }
              sample.w = 1.f;

              if (!isfinite(grad.x) || !isfinite(grad.y) || !isfinite(grad.z)) {
                grad = vec3f(0.f);
              }

              if (length(grad) > .0f) {
                const vec3f lightDir = -vec3f(ray.direction);
                const float scale
                  = .3f + .7f*fabsf(dot(lightDir,grad))
                  / sqrtf(dot(grad,grad));
                (vec3f&)sample *= scale;
              }
              result.pixelColor += (1.f-result.pixelColor.w)*sample.w*vec4f(vec3f(sample),1.f);
              result.t_hit = tavg;
              result.gradient = grad;
            }
        }
      }

      last_t=t_sample;
      lastCellValue=cellValue;
      lastGradient=gradient;
    }

    float last_t;
    float lastCellValue;
    vec3f lastGradient;
  };

  template<bool GRADIENT_SHADING>
  inline __device__
  void integrateBrick(IntegrationResult &result,
                      const float interleavedSamplingOffset,
                      const owl::Ray &ray,
                      int leafID,
                      float t0, float t1,
                      const int numChannels,
                      int &steps)
  {
    const float global_dt = optixLaunchParams.dt;
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = optixLaunchParams.sameBrickRegionsBuffer[leafID];
    const float dt = global_dt * region.finestLevelCellWidth;
    // TODO: if we finally drop the "not PROPER_xxx" code, we
    // no longer need to pass this on the integration function!
    const int finestLevelCellWidth = (int)region.finestLevelCellWidth;
#else
    const Brick &brick = optixLaunchParams.brickBuffer[leafID];
    const float dt = global_dt * (1<<brick.level);
    // TODO: if we finally drop the "not PROPER_xxx" code, we
    // no longer need to pass this on the integration function!
    const int finestLevelCellWidth = 1;
#endif

    int i0 = int(ceilf((t0-dt*interleavedSamplingOffset) / dt));
    float t_i = (interleavedSamplingOffset + i0) * dt;
    while ((t_i-dt) >= t0) t_i = t_i-dt;
    while (t_i < t0) t_i += dt;

# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
    float t_last = t0;
# endif    

# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
    for (;true;t_i += dt)
#else
      for (;t_i <= t1; t_i += dt)
#endif
        {
        
# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
          const float t_next = min(t_i,t1);
          const float t_sample = 0.5f*(min(t1,t_next)+t_last);
          const float actual_dt = t_next-t_last;
          t_last = t_next;
# else
          const float t_sample = t_i;
          const float actual_dt = dt;
# endif    
          const vec3f pos = ray.origin + t_sample * ray.direction;
          float cellValue = 0.f;
          vec3f grad(0.f);
          for (int c=0; c<numChannels; ++c) {
            if (GRADIENT_SHADING) {
              if (samplePointWithDerivative<false>(cellValue, grad, leafID, pos, c)) {
                integrateVolume(ray,result,t_sample,actual_dt,cellValue,grad,finestLevelCellWidth,leafID,c);
              }
            } else {
              if (samplePoint(cellValue, leafID, pos, c)) {
                integrateVolume(ray,result,t_sample,actual_dt,cellValue,grad,finestLevelCellWidth,leafID,c);
              }
            }
          }
          if (result.pixelColor.w >= TERMINATION_THRESHOLD) break;
# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
          if (t_next >= t1) break;
# endif
        }
  }
  
  inline __device__
  void isoIntegrateBrick(IsoSurfaceIntegrationFunction* integrationFuncs,
                         IntegrationResult &result,
                         const float interleavedSamplingOffset,
                         const owl::Ray &ray,
                         int leafID,
                         float t0, float t1,
                         const int numChannels,
                         int &steps)
  {
    const float global_dt = optixLaunchParams.dt;
#if EXPLICIT_BASIS_METHOD
    const SameBricksRegion &region = optixLaunchParams.sameBrickRegionsBuffer[leafID];
    const float dt = global_dt * region.finestLevelCellWidth;
    // TODO: if we finally drop the "not PROPER_xxx" code, we
    // no longer need to pass this on the integration function!
    const int finestLevelCellWidth = (int)region.finestLevelCellWidth;
#else
    const Brick &brick = optixLaunchParams.brickBuffer[leafID];
    const float dt = global_dt * (1<<brick.level);
    // TODO: if we finally drop the "not PROPER_xxx" code, we
    // no longer need to pass this on the integration function!
    const int finestLevelCellWidth = 1;
#endif

    int i0 = int(ceilf((t0-dt*interleavedSamplingOffset) / dt));
    float t_i = (interleavedSamplingOffset + i0) * dt;
    while ((t_i-dt) >= t0) t_i = t_i-dt;
    while (t_i < t0) t_i += dt;

# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
    float t_last = t0;
# endif    

# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
    for (;true;t_i += dt)
#else
      for (;t_i <= t1; t_i += dt)
#endif
        {
        
# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
          const float t_next = min(t_i,t1);
          const float t_sample = 0.5f*(min(t1,t_next)+t_last);
          const float actual_dt = t_next-t_last;
          t_last = t_next;
# else
          const float t_sample = t_i;
          const float actual_dt = dt;
# endif    
          const vec3f pos = ray.origin + t_sample * ray.direction;
          for (int c=0; c<numChannels; ++c) {
            bool doIntegrate = false;
            float cellValue = 0.f;
            vec3f grad(0.f);
            if (optixLaunchParams.gradientShadingISO)
              doIntegrate = samplePointWithDerivative<false>(cellValue, grad, leafID, pos, c);
            else
              doIntegrate = samplePoint(cellValue, leafID, pos, c);

            if (doIntegrate) {
              integrationFuncs[c](ray,result,t_sample,actual_dt,cellValue,grad,finestLevelCellWidth,leafID,c);
              if (result.pixelColor.w >= TERMINATION_THRESHOLD) break;
            }
          }
# if PROPER_DISTANCE_IN_OPACITY_CORRECTION
          if (t_next >= t1) break;
# endif
        }
  }
  
  inline __device__
  void clipRay(owl::Ray &ray)
  {
    const FrameState &global = optixLaunchParams.frameStateBuffer[0];
    if (!global.clipBox.enabled) return;
    
    boxTest(ray,global.clipBox.coords,ray.tmin,ray.tmax);
  }
 
  inline __device__
  float intersectLinePlane(const vec3f &p1,
                           const vec3f &p2,
                           const vec3f &normal,
                           float offset)
  {
    float s = dot(normal, normalize(p2-p1));

    if (s == 0.f)
      return -1.f;

    float t = (offset - dot(normal, p1)) / s;

    if (t < 0.f || t > length(p2-p1))
      return -1.f;

    return t;
  }


  inline __device__
  void intersectBoxPlane(const box3f &box,
                         const vec3f &normal,
                         float offset,
                         vec3f *pts,
                         int &isectCnt)
  {
    const int key[4][3] = { {0,0,0}, {1,0,1}, {1,1,0}, {0,1,1} };
    vec3f corners[2] = { box.lower, box.upper };
    isectCnt = 0;

    for (int i=0; i<4 && isectCnt<6; ++i) {
      for (int j=0; j<3 && isectCnt<6; ++j) {
        // Edge to intersect with
        vec3f p1((j==0) ? (corners[1-key[i][0]][0]) : corners[key[i][0]][0],
                 (j==1) ? (corners[1-key[i][1]][1]) : corners[key[i][1]][1],
                 (j==2) ? (corners[1-key[i][2]][2]) : corners[key[i][2]][2]);
        vec3f p2(corners[key[i][0]][0], corners[key[i][1]][1], corners[key[i][2]][2]);

        // Intersect edge with plane
        float t = intersectLinePlane(p1,p2,normal,offset);

        if (t >= 0.f) {
          pts[isectCnt++] = p1 + normalize(p2-p1) * t;
        }
      }
    }
  }

  inline __device__
  float intersectRayTriangle(owl::Ray ray,
                             const vec3f &v1,
                             const vec3f &e1,
                             const vec3f &e2)
  {
    vec3f s1 = cross(ray.direction, e2);
    float div = dot(s1, e1);

    if (div == 0.f)
      return -1.f;

    float invDiv = 1.f / div;

    vec3f d = ray.origin - v1;
    float b1 = dot(d, s1) * invDiv;

    if (b1 < 0.f || b1 > 1.f)
      return -1.f;

    vec3f s2 = cross(d, e1);
    float b2 = dot(ray.direction, s2) *invDiv;

    if (b2 < 0.f || b1 + b2 > 1.f)
      return -1.f;

    return dot(e2, s2) * invDiv;
  }

  inline __device__
  SurfacePRD traceContourRay(owl::Ray ray,
                             const vec3f &normal,
                             float offset,
                             int channel)
  {
    SurfacePRD prd;
    prd.t_hit = ray.tmax;

    vec3f pts[6];
    int isectCnt;
    intersectBoxPlane(box3f{{0.f,0.f,0.f},{1.f,1.f,1.f}},normal,offset,pts,isectCnt);

    // Scale to world bounds
    for (int i=0; i<isectCnt; ++i) {
      pts[i] *= optixLaunchParams.worldSpaceBounds_hi-optixLaunchParams.worldSpaceBounds_lo;
      pts[i] += optixLaunchParams.worldSpaceBounds_lo;
    }

    float t = -1.f;

    // Cyclical selection-sort
    for (int i=0; i<isectCnt-1; ++i)
      {
        int minIdx = i;  
        for (int j=i+1; j<isectCnt; ++j) {
          vec3f p1 = pts[j];
          vec3f p2 = pts[minIdx];
          vec3f v = cross(p1-pts[0],p2-pts[0]);
          if (dot(v, normal) < 0.f)
            minIdx = j;
        }

        // swap
        vec3f tmp = pts[i];
        pts[i] = pts[minIdx];
        pts[minIdx] = tmp;
      }  

    for (int i=2; i<isectCnt; ++i) {
      vec3f v1 = pts[0];
      vec3f e1 = pts[i-1]-v1;
      vec3f e2 = pts[i]-v1;

      float tt = intersectRayTriangle(ray,v1,e1,e2);
      if (tt >= 0.f && (tt < t || t < 0.f)) t = tt;
    }

    if (t < 0.f)
      return prd;

    float value = samplePointWithInfRay(ray.origin+ray.direction*t, 0);
    vec4f sample = lookupTransferFunction(optixLaunchParams.frameStateBuffer[0], value, channel);

    prd.primID = PRIMID_PLANE;
    prd.t_hit = t;
    prd.Ng = normal;
    prd.ambient = 0.f;
    prd.baseColor = vec3f(sample.x,sample.y,sample.z);

    return prd;
  }

  inline __device__
  SurfacePRD traceIsoRay(owl::Ray ray,
                         float interleavedSamplingOffset)
  {
    int steps = 0;
    // first, since we now traverse bricks and sample cells: convert ray to voxel space...
    const FrameState &fs = optixLaunchParams.frameStateBuffer[0];

    ray.origin = xfmPoint(fs.voxelSpaceTransform,ray.origin);
    ray.direction = xfmVector(fs.voxelSpaceTransform,ray.direction);

    const float dt_scale = length(vec3f(ray.direction));
    ray.direction = normalize(vec3f(ray.direction));

    float alreadyIntegratedDistance = dt_scale * ray.tmin;//0.f;

    IsoSurfaceIntegrationFunction isoSurfaceIntegrationFuncs[MAX_CHANNELS];

    SurfacePRD result;
    result.t_hit = ray.tmax;

    while (1) {
      IsoSurfacePRD prd;
      prd.leafID = -1;
      prd.t0 = prd.t1 = 0.f; // doesn't matter as long as leafID==-1
      ray.tmin = alreadyIntegratedDistance;
      ray.tmax = ray.tmax * dt_scale;
      owl::traceRay(optixLaunchParams.isoSurfaceBVH, ray, prd,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      if (prd.leafID < 0)
        break;

      vec4f pixelColor(0.f);
      IntegrationResult intResult(pixelColor);
      isoIntegrateBrick(isoSurfaceIntegrationFuncs,intResult,
                        interleavedSamplingOffset,
                        ray,prd.leafID,max(ray.tmin,prd.t0),min(ray.tmax,prd.t1),
                        optixLaunchParams.numPrimaryChannels,
                        steps);
      if (intResult.t_hit >= 0.f) {
        // Report a surface intersection if we're fully opaque
        result.primID = PRIMID_ISOSURFACE; 
        // Back from voxel to ray space...
        result.t_hit = intResult.t_hit/dt_scale;
        result.Ng = normalize(intResult.gradient);
        result.ambient = 0.f;
        result.baseColor = vec3f(intResult.pixelColor);
        return result;
      }
      alreadyIntegratedDistance = prd.t1 * (1.0000001f);
    }
    return result;
  }

  inline __device__ bool isIsoSurfaceID(int primID)   { return primID == PRIMID_ISOSURFACE; /*TODO!?!?!*/ }
  inline __device__ bool isContourPlaneID(int primID) { return primID == PRIMID_PLANE; /*TODO!?!?!*/ }
  inline __device__ bool isStreamlineID(int primID)   { return primID == PRIMID_STREAMLINE; /*TODO!?!?!*/ }
  inline __device__ bool isVisSurfaceID(int primID)   { return isIsoSurfaceID(primID) || isContourPlaneID(primID) || isStreamlineID(primID); }

  enum SurfaceTypeFlags {
                         ST_MESHES = 0x1,
                         ST_CONTOUR_PLANES = 0x2,
                         ST_ISO_SURFACES = 0x4,
                         ST_STREAMLINES = 0x8,
                         ST_ALL_SURFACES = 0xffffffff
  };

  inline __device__
  void traceSurfaces(owl::Ray ray,
                     SurfacePRD &prd,
                     SurfaceTypeFlags surfaceTypes)
  {
    prd.primID = -1;
    prd.t_hit  = ray.tmax;
    
    // First trace triangle geo
    if (surfaceTypes & ST_MESHES)
      owl::traceRay(optixLaunchParams.surfaceModel, ray, prd,
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT);
  
    // Then contour planes
    if (surfaceTypes & ST_CONTOUR_PLANES)
      {
        for (int i=0;i<MAX_CONTOUR_PLANES;++i) {
          if (optixLaunchParams.frameStateBuffer[0].contourPlane[i].enabled) {
            SurfacePRD contourPRD = traceContourRay(ray,
                                                    optixLaunchParams.frameStateBuffer[0].contourPlane[i].normal,
                                                    optixLaunchParams.frameStateBuffer[0].contourPlane[i].offset,
                                                    optixLaunchParams.frameStateBuffer[0].contourPlane[i].channel);
            if (isContourPlaneID(contourPRD.primID) && contourPRD.t_hit < prd.t_hit)
              prd = contourPRD;
          }
        }
      }

    if (surfaceTypes & ST_STREAMLINES)
      {
        SurfacePRD streamlinePRD;
        streamlinePRD.primID = -1;
        streamlinePRD.t_hit = 2e10f;
        owl::traceRay(optixLaunchParams.streamlineBVH, ray, streamlinePRD,
                      OPTIX_RAY_FLAG_DISABLE_ANYHIT);
        if (isStreamlineID(streamlinePRD.primID) && streamlinePRD.t_hit < prd.t_hit)
          prd = streamlinePRD;
      }

#if ISO_SURFACES
    // Now trace iso surfaces
    if (surfaceTypes & ST_ISO_SURFACES)
      {
        bool activeIsoSurfaces = false;
        for (int i=0;i<MAX_ISO_SURFACES;i++)
          activeIsoSurfaces |= optixLaunchParams.frameStateBuffer[0].isoSurface[i].enabled;

        if (activeIsoSurfaces) {
          SurfacePRD isoPRD = traceIsoRay(ray,0.f);
          if (isIsoSurfaceID(isoPRD.primID) && isoPRD.t_hit < prd.t_hit)
            prd = isoPRD;
        }
      }
#endif
  }

  inline __device__
  void computeTraces()
  {
    const vec2i pixelID = owl::getLaunchIndex();
    const vec2i launchDim = owl::getLaunchDims();
    const int pixelIdx = pixelID.x+optixLaunchParams.fbSize.x*pixelID.y;
    const int t = optixLaunchParams.currentTimestep[0];

    if (t < optixLaunchParams.numTimesteps && pixelIdx < optixLaunchParams.numTraces && optixLaunchParams.traces != nullptr) {
      box3f bbox(optixLaunchParams.worldSpaceBounds_lo,
                 optixLaunchParams.worldSpaceBounds_hi);
      int i = pixelIdx;
      vec3f p = optixLaunchParams.traces[i*optixLaunchParams.numTimesteps+(t-1)];
      vec3f pp = p;
      if (p.x < 2e10f) {
        bool valid = true;

        // Runge-Kutta algorithm
        vec3f k1;
        valid &= sampleDirection(p,k1);
        k1 *= optixLaunchParams.steplen;
        vec3f ptry1 = p + k1 * .5f;

        vec3f k2;
        valid &= sampleDirection(ptry1,k2);
        k2 *= optixLaunchParams.steplen;
        vec3f ptry2 = p + k2 * .5f;

        vec3f k3;
        valid &= sampleDirection(ptry2,k3);
        k3 *= optixLaunchParams.steplen;
        vec3f ptry3 = p + k3;

        vec3f k4;
        valid &= sampleDirection(ptry3,k4);
        k4 *= optixLaunchParams.steplen;
        p += 1/6.f * (k1 +2.f*k2 +2.f*k3 + k4);

        if (!valid || !bbox.contains(p) || length(p-pp) < 1e-10f)
          p = vec3f(2e10f);
      }
      optixLaunchParams.traces[i*optixLaunchParams.numTimesteps+t] = p;
    }
  }

  OPTIX_RAYGEN_PROGRAM(renderFrame)()
  {
    const vec2i pixelID = owl::getLaunchIndex();
    const vec2i launchDim = owl::getLaunchDims();

    if (optixLaunchParams.tracerEnabled)
      computeTraces();

    if (pixelID.x >= optixLaunchParams.fbSize.x) return;
    if (pixelID.y >= optixLaunchParams.fbSize.y) return;
    const int pixelIdx = pixelID.x+optixLaunchParams.fbSize.x*pixelID.y;

    uint64_t clockBegin = clock();
    const FrameState &global = optixLaunchParams.frameStateBuffer[0];
    const int frameID = global.frameID;
    Random rnd(frameID*launchDim.x*launchDim.y+
               (unsigned)pixelID.x,(unsigned)pixelID.y);

    vec2f pixelSample = vec2f(pixelID) + vec2f(rnd(),rnd());
    owl::Ray ray = Camera::generateRay(global, pixelSample, rnd);

    SurfacePRD surface;
    surface.primID = -1;
    surface.t_hit  = ray.tmax;
   
    traceSurfaces(ray, surface, ST_ALL_SURFACES);

    vec3f bgColor = vec3f(0.f);
    if (surface.primID >= 0 || isVisSurfaceID(surface.primID)) {

      const bool shade = surface.primID >= 0
        || isStreamlineID(surface.primID)
        || isContourPlaneID(surface.primID)
        || (isIsoSurfaceID(surface.primID) && optixLaunchParams.gradientShadingISO);

      if (shade && length(surface.Ng) > 0.f) {
        // AO
        const float AO_Radius = global.ao.length;
        const int AO_Samples = global.ao.enabled ? 2 : 0;

        vec3f isect_pos = vec3f(ray.origin)+vec3f(ray.direction)*surface.t_hit;
        vec3f u;
        vec3f v;
        vec3f w = surface.Ng;
        make_orthonormal_basis(u,v,w);

        int hitCnt = 0;
        for (int i=0; i<AO_Samples; ++i) {
          vec3f sp = cosine_sample_hemisphere(rnd(),rnd());

          vec3f dir = normalize(sp.x*u + sp.y*v + sp.z*w);

          owl::Ray ao_ray((float3)isect_pos,
                          (float3)dir,
                          /* tmin     : */ 1e-4f,
                          /* tmax     : */ AO_Radius);

          SurfacePRD ao;
          ao.primID = -1;
          ao.t_hit = ray.tmax;

          traceSurfaces(ao_ray, ao,
                        (SurfaceTypeFlags)(ST_ALL_SURFACES & ~ST_CONTOUR_PLANES)
                        );
          if (ao.primID >= 0 || isVisSurfaceID(ao.primID)) {
            hitCnt++;
          }
        }

        float shadow = global.ao.enabled ? (float)hitCnt/AO_Samples : 0.f;
        bgColor
          = surface.ambient
          + surface.baseColor*fabs(dot(vec3f(ray.direction),surface.Ng))*(1.f-shadow);
      } else {
        bgColor = surface.baseColor;
      }
    }
    
    vec4f pixelColor(0.f);
    float interleavedSamplingOffset = rnd();//0.f;

    ray.tmax = surface.t_hit;
    clipRay(ray);
    surface.t_hit = ray.tmax;

    int steps = 0;
    // first, since we now traverse bricks and sample cells: convert ray to voxel space...
    const FrameState &fs = optixLaunchParams.frameStateBuffer[0];
    ray.origin = xfmPoint(fs.voxelSpaceTransform,ray.origin);
    ray.direction = xfmVector(fs.voxelSpaceTransform,ray.direction);

    const float dt_scale = length(vec3f(ray.direction));
    ray.direction = normalize(vec3f(ray.direction));

    pixelColor = vec4f(0.f);

    float alreadyIntegratedDistance = dt_scale * ray.tmin;//0.f;
    // float last_t = -1.f;
    // bool funny = false;
    while (1) {
      ray.tmin = alreadyIntegratedDistance;
      ray.tmax = surface.t_hit * dt_scale;
      VolumePRD prd = traceVolumeRay(ray);
      if (prd.leafID < 0)
        break;

      IntegrationResult intResult(pixelColor);
      if (optixLaunchParams.gradientShadingDVR) {
        integrateBrick<true>(intResult,interleavedSamplingOffset,
                             ray,prd.leafID,prd.t0,prd.t1,
                             optixLaunchParams.numPrimaryChannels,
                             steps);
      } else {
        integrateBrick<false>(intResult,interleavedSamplingOffset,
                              ray,prd.leafID,prd.t0,prd.t1,
                              optixLaunchParams.numPrimaryChannels,
                              steps);
      }
      if (pixelColor.w >= TERMINATION_THRESHOLD) {
        pixelColor = vec4f(vec3f(pixelColor)*pixelColor.w,1.f);
        break;
      }
      alreadyIntegratedDistance = prd.t1 * (1.0000001f);
    }
    
    vec3f color = pixelColor.w*vec3f(pixelColor) + (1.f-pixelColor.w)*bgColor;
    
    if (global.clockScale > 0.f) {
      uint64_t absClock = clock()-clockBegin;
      float relClock = global.clockScale * absClock/1000000.f;
      color.x = min(relClock,1.f);
    }
    
    if (frameID > 0)
      color += vec3f(optixLaunchParams.accumBufferPtr[pixelIdx]);

    optixLaunchParams.accumBufferPtr[pixelIdx] = vec4f(color,1.f);

    color = color / (frameID + 1.f);
    color.x = linear_to_srgb(color.x);
    color.y = linear_to_srgb(color.y);
    color.z = linear_to_srgb(color.z);

    optixLaunchParams.colorBufferPtr[pixelIdx] = make_rgba8(color);
  }

  /* set up empty miss program */
  OPTIX_MISS_PROGRAM(missProg)() {}
}

