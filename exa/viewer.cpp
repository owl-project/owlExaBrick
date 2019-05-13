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

#include <array>
#include <numeric>
#include <sstream>
#include "common.h"
#include "owl/owl.h"
#include <GL/glut.h>
#include <GL/freeglut.h>
// optix
#include "programs/FrameState.h"
#include "exa/OptixRenderer.h"
#include "exa/ColorMapper.h"
#include "exa/Config.h"
#include <GL/glui.h>
#include <GL/glui/TransferFunction.h>
// stb image, for screen shot
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION 1
#include "submodules/3rdParty/stb_image.h"
#include "embedded_colormaps.h"

#include "../glutViewer/OWLViewer.h"

inline void stringReplace(std::string& str, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), to);
    pos += to.length();
  }
};

namespace exa {

  OptixRenderer::SP renderer;
  struct MyRootWindow;
  std::shared_ptr<MyRootWindow> viewer;
  
  using namespace glui;
  
  static const int XF_ALPHA_COUNT = exa::NUM_XF_VALUES; //128;
  
  struct {
    std::vector<std::array<float, XF_ALPHA_COUNT>> xfAlpha;
    
    struct {
      vec3f vp = vec3f(0.f);
      vec3f vu = vec3f(0.f);
      vec3f vi = vec3f(0.f);
      float fov = 70.f;
    } camera;
    vec2i windowSize = vec2i(0.f);

    std::string displayString;

    struct {
      float length  = 1000.f;
      int   enabled = false;
    } ao;

    float clockScale = 0.f;
    int doProgressiveRefinement { 1 };
    int gradientShadingDVR { 1 };
    int gradientShadingISO { 1 };
    int showColorBar { 0 };
    int colorBarChannel { -1 };
    
    struct {
      box3f coords = box3f(vec3f(0),vec3f(1));
      // float orientation[4][4] = { {1,0,0,0},  {0,1,0,0},  {0,0,1,0},  {0,0,0,1} };
      // vec3f origin { vec3f(.5f) };
      int   enabled { 0 };
      // int   inverted { 0 };
    } clipBox;

    struct {
      vec3i channels { 0,1,2 };
      box3f seedRegion {{.3f,.3f,.5f},{.8f,.8f,.5f}};
      int numTraces = 1000;
      int numTimesteps = 100;
      float steplen = 1e-6f;
      int enabled { 0 };
    } traces;

    interval<float> valueRange = interval<float>(1e20f, -1e20f);

    std::vector<float> isovalues;
    std::vector<int> isochannels;
    std::vector<vec4f> contourplanes; // normal (xyz), offset (w)
    std::vector<int> contourchannels;
    std::vector<std::string> colormaps;
    std::string customColorMapString;
    int orbitCount = 0;
    vec3f orbitUp = vec3f(0, 1, 0);
    vec3f orbitCenter = vec3f(1e20f);
    float orbitRadius = -1.f;

    float xfOpacityScale = 1.f;

    float dt = .5f;
  } cmdline;
 
  struct MyRootWindow : public owl::glutViewer::OWLViewer
  {
    typedef std::shared_ptr<MyRootWindow> SP;
    typedef owl::glutViewer::OWLViewer inherited;

    MyRootWindow(int argc, char** argv, const box3f &bounds)
      : OWLViewer(argc,argv,"ExaBricks",{600,400}), bounds(bounds)
    {
      gradientShadingDVR = cmdline.gradientShadingDVR;
      gradientShadingISO = cmdline.gradientShadingISO;
      renderer->setGradientShadingDVR(gradientShadingDVR);
      renderer->setGradientShadingISO(gradientShadingISO);
    }
    
    /*! this gets called when the user presses a key on the keyboard ... */
    void key(char key, const vec2i &where) override
    {
      switch (key) {
      case 't': {
        renderer->setTracerEnabled(!renderer->traces.tracerEnabled);
      } break;
      case 'T': {
        std::string fileName = "currentTransferFunction.xf";
        std::cout << "#viewer: Dumping transfer function (opacities, channel " << currentChannel << ") to '" << fileName << "'" << std::endl;
        std::ofstream xfFile(fileName,std::ios::binary);
        xfFile.write((char*)cmdline.xfAlpha[currentChannel].data(),XF_ALPHA_COUNT*sizeof(float));
        std::cout << "#viewer: done writing transfer function (opacities) to '" << fileName << "'" << std::endl;
      } break;
      case 'C': {
        auto &fc = camera;
        std::cout << "(C)urrent camera:" << std::endl;
        std::cout << "- from :" << fc.position << std::endl;
        std::cout << "- poi  :" << fc.getPOI() << std::endl;
        std::cout << "- upVec:" << fc.upVector << std::endl; 
        std::cout << "- frame:" << fc.frame << std::endl;
        std::cout.precision(10);
        std::cout << "cmdline: --camera "
                  << fc.position.x << " "
                  << fc.position.y << " "
                  << fc.position.z << " "
                  << fc.getPOI().x << " "
                  << fc.getPOI().y << " "
                  << fc.getPOI().z << " "
                  << fc.upVector.x << " "
                  << fc.upVector.y << " "
                  << fc.upVector.z << " "
                  << "--fov " << cmdline.camera.fov
                  << std::endl;
      } break;
      case '!': {
        screenShotGL();
      } break;
      default:
        inherited::key(key,where);
      }
    }
    
    /*! saves a screenshot in 'tetty.png' */
    void screenShot(std::string fileName = "screenshot.png")
    {
      const uint32_t *fb = (const uint32_t *)fbPointer;
      const vec2i fbSize = renderer->fbSize;

      // auto now = std::chrono::system_clock::now();
      // auto in_time_t = std::chrono::system_clock::to_time_t(now);
      // struct tm *tm = localtime(&in_time_t);

      // char format[100];
      // sprintf(format,"%04d-%02d-%02d_%02dh%02d",
      //         tm->tm_year+1900,
      //         tm->tm_mon,
      //         tm->tm_mday,
      //         tm->tm_hour,
      //         tm->tm_min);
      std::vector<uint32_t> pixels;
      for (int y=0;y<fbSize.y;y++) {
        const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
        for (int x=0;x<fbSize.x;x++) {
          pixels.push_back(line[x]);
        }
      }

      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     pixels.data(),fbSize.x*sizeof(uint32_t));
      std::cout << "screenshot saved in '" << fileName << "'" << std::endl;
    }

    void screenShotGL(std::string fileName = "screenshot.png")
    {
      const vec2i fbSize = renderer->fbSize;
      std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
      glReadPixels(0,0,fbSize.x,fbSize.y,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());

      // swap rows
      for (int y=0;y<fbSize.y/2;y++) {
        for (int x=0;x<fbSize.x;x++) {
          std::swap(pixels[y*fbSize.x+x],pixels[(fbSize.y-y-1)*fbSize.x+x]);
        }
      }

      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     pixels.data(),fbSize.x*sizeof(uint32_t));
      std::cout << "screenshot saved in '" << fileName << "'" << std::endl;
    }

      
    /*! this function gets called whenever any camera manipulator
      updates the camera. gets called AFTER all values have been updated */
    virtual void cameraChanged() override
    {
      const owl::glutViewer::SimpleCamera camera = inherited::getSimplifiedCamera();

      const vec3f screen_du = camera.screen.horizontal / float(getWindowSize().x);
      const vec3f screen_dv = camera.screen.vertical   / float(getWindowSize().y);

      resetAccumulation();
      renderer->updateCamera(camera.lens.center,
                             camera.screen.lower_left,
                             screen_du,
                             screen_dv);
    }

    void advanceOrbit()
    {
        ++currentOrbitPos;
        vec3f nextPos = cmdline.orbitCenter + orbitCameraPositions[currentOrbitPos];
        nextPos.z = std::abs(nextPos.z);
        std::cout << "Advance to orbit #" << currentOrbitPos << "\nOrbit pos " << nextPos << "\n";
        camera.setOrientation(nextPos,
                cmdline.orbitCenter,
                cmdline.orbitUp,
                cmdline.camera.fov);
        updateCamera();
    }

    /*! tell renderer to re-start frame accumulatoin; this _has_ to be
        called every time something changes that would change the
        converged image to be rendererd (eg, change to transfer
        runctoin, camera, etc) */
    void resetAccumulation()
    {
      accumID = 0;
    }
  
    template <class T>
    inline T lerp(const T& a, const T& b, float x)
    {
      return (1.0f - x) * a + x * b;
    };

    inline vec3f randomColor(size_t idx)
    {
        unsigned int r = (unsigned int)(idx*13*17 + 0x234235);
        unsigned int g = (unsigned int)(idx*7*3*5 + 0x773477);
        unsigned int b = (unsigned int)(idx*11*19 + 0x223766);
        return vec3f((r&255)/255.f,
                (g&255)/255.f,
                (b&255)/255.f);
    }


    virtual void render() override
    {
      renderer->updateDt(cmdline.dt);
      renderer->updateFrameID(accumID);
      if (renderer->traces.tracerEnabled)
        if (renderer->advanceTracer()) {}//resetAccumulation();
      if (cmdline.doProgressiveRefinement)
        ++accumID;
      
      renderer->render();
      
      if (!textureID)
        glGenTextures(1,&textureID);

      uint32_t *fb = (uint32_t *)owlBufferGetPointer(renderer->colorBuffer, 0);

      glutSwapBuffers();

      static double t_last = 0.0;
      static size_t frame_id = 1;
      bool resetBenchmark = false;
      double t_now = getCurrentTime();
      if (t_last != 0.) {
        double thisFPS = 1./(t_now-t_last);
        if (fps == 0.f)
          fps = thisFPS;
        else
          fps += thisFPS;
        char newTitle[128] = {0};
        sprintf(newTitle,"exaBricks (%3.2ffps)",fps / frame_id);
        if (frame_id == 50) {
            std::cout << "Avg. after " << frame_id
                << " frames: " << fps / frame_id << " FPS (" << 1000.f / (fps / frame_id) << "ms)\n" << std::flush;
            char fname[512] = {0};
            static int outputImgId = 0;
            std::snprintf(fname, 511, "bench_screenshot%08d.png", outputImgId);
            ++outputImgId;
            screenShotGL(fname);
            if (!orbitCameraPositions.empty()) {
                if (currentOrbitPos + 1 < orbitCameraPositions.size()) {
                    advanceOrbit();
                    frame_id = 1;
                    t_last = 0.0;
                    fps = 0;
                    resetBenchmark = true;
                } else {
                    std::cout << "All camera positions rendered\n";
                }
            }
            //exit(0);
        }
        setTitle(newTitle);
        if (!resetBenchmark) {
            ++frame_id;
        }
      }

      if (!resetBenchmark) {
        t_last = t_now;
      }
    }

    virtual void draw() override
    {
      glPushAttrib(GL_ALL_ATTRIB_BITS);

      inherited::draw();

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);

      // Draw overlay
      if (!cmdline.displayString.empty()) {
        glColor3f(1.f,1.f,1.f);
        glRasterPos2f(.03f, .97f);
        glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)cmdline.displayString.c_str());
      }

      if (cmdline.showColorBar && customColorMap != nullptr) {
        // Local copy that is guaranteed to have 0.f and 1.f values!
        ColorMapper cm = *customColorMap;
        if (cm.values_.front().first != 0.f) {
          auto cp = cm.values_.front();
          cp.first = 0.f;
          cm.values_.insert(cm.values_.begin(),cp);
        }

        if (cm.values_.back().first != 1.f) {
          auto cp = cm.values_.back();
          cp.first = 1.f;
          cm.values_.insert(cm.values_.end(),cp);
        }

        for (unsigned i=0; i != cm.values_.size() - 1; ++i)
        {
          int chan=cmdline.colorBarChannel >= 0 ? cmdline.colorBarChannel : currentChannel;
          auto cp1 = cm.values_[i];
          auto cp2 = cm.values_[i+1];
          //std::cout << cp.first << '\n';
          float y1 = cp1.first * 2.f - 1.f;
          float y2 = cp2.first * 2.f - 1.f;

          float value = cp2.first;
          value *= xfDomain[chan].upper;
          std::stringstream str;
          str << std::fixed;
          //str << std::scientific;
          str.precision(3);
          str << value;
          //value /= xfDomain[4].upper;
          std::string sval = str.str();

          glColor3f(1.f,1.f,1.f);
          glRasterPos2f(.87f, y2);
          glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)sval.c_str());

          if (i == cm.values_.size() - 2) {
            auto cp = cm.values_.back();
            float y = cp.first * 2.f - 1.f;
            float value = cp.first;
            value *= xfDomain[chan].upper;
            std::stringstream str;
            str << std::fixed;
            //str << std::scientific;
            str.precision(3);
            str << value;
            //value /= xfDomain[4].upper;
            std::string sval = str.str();

            glColor3f(1.f,1.f,1.f);
            glRasterPos2f(.87f, y-.05f);
            glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)sval.c_str());
          }

          vec3f rgb1 = cp1.second;
          vec3f rgb2 = cp2.second;

          glBegin(GL_QUADS);
            glColor3f(rgb1.x,rgb1.y,rgb1.z);
            glVertex3f(.97f,y1,0.f);

            glColor3f(rgb2.x,rgb2.y,rgb2.z);
            glVertex3f(.97f,y2,0.f);

            glColor3f(rgb2.x,rgb2.y,rgb2.z);
            glVertex3f(1.f,y2,0.f);

            glColor3f(rgb1.x,rgb1.y,rgb1.z);
            glVertex3f(1.f,y1,0.f);
          glEnd();
        }
      }

      glPopAttrib();
    }

    virtual void resize(const vec2i &newSize) override
    {
      inherited::resize(newSize);
      renderer->resizeFrameBuffer(fbPointer,newSize);
      float aspect = newSize.x/float(newSize.y);
      inherited::setAspect(aspect);
      inherited::updateCamera();
      owlBuildSBT(renderer->context);
    }


    ColorMapper* customColorMap = nullptr;
    std::vector<interval<float>> xfDomain;
    
    const box3f bounds;
    std::vector<vec3f> orbitCameraPositions;
    size_t currentOrbitPos = 0;

    int currentChannel = 0;
    
    float fps = 0.f;
    /*! accumulation ID */
    int accumID { 0 };
    /*! whether to progressively refine - must be an int to match the glui data type */
    int doSpaceSkipping { 1 };
    int gradientShadingDVR { 1 };
    int gradientShadingISO { 1 };
    GLuint textureID { 0 };
    // imglui::TransferFunctionWidget::SP xfWidget;
  };






  /*! glui-based user interface - since glui can't save actual context
    objects we have to use a global 'current' here, and can only
    ever have one of those active at any time */
  struct GLUIInterface {
    static GLUIInterface *current;

    GLUIInterface(int glutWindowHandle)
      : xfColorMap(renderer->scalarFields.size())
      , xfDomain(renderer->scalarFields.size())
    {
      // std::vector<std::string> shadeModes = renderer.getSupportedShadeModes();
      
      current = this;
      context = glui::GLUI_Master.create_glui( "ExaBricks GLUI", 0); 
      context->set_main_gfx_window(glutWindowHandle);

      const int invalidID = -1;
      
      // glui::GLUI_Panel *shadeModePanel = new glui::GLUI_Panel( context, "Render Mode" );
      // auto shadeModeRadio = new glui::GLUI_RadioGroup(shadeModePanel,&shadeMode,
      //                                                 invalidID,shadeModeCB);
      // for (auto rm : shadeModes)
      //   new glui::GLUI_RadioButton( shadeModeRadio, rm.c_str() );

      // ==================================================================
      // create some color maps
      // ==================================================================

      // Embedded colormaps
      const static std::vector<std::string> colormapNames = {
          "Paraview Cool Warm",
          "Rainbow",
          "Matplotlib Plasma",
          "Matplotlib Virdis",
          "Samsel Linear Green",
          "Samsel Linear YGB 1211g",
          "Cool Warm Extended",
          "Blackbody",
          "Jet",
          "Blue Gold",
          "Ice Fire",
          "Nic Edge",
          "Covise",
          "JamieDraft",
          "HSV",
          "Custom"
      };

      std::vector<ColorMapper> colormaps = {
          ColorMapper(paraview_cool_warm, sizeof(paraview_cool_warm)),
          ColorMapper(rainbow, sizeof(rainbow)),
          ColorMapper(matplotlib_plasma, sizeof(matplotlib_plasma)),
          ColorMapper(matplotlib_virdis, sizeof(matplotlib_virdis)),
          ColorMapper(samsel_linear_green, sizeof(samsel_linear_green)),
          ColorMapper(samsel_linear_ygb_1211g, sizeof(samsel_linear_ygb_1211g)),
          ColorMapper(cool_warm_extended, sizeof(cool_warm_extended)),
          ColorMapper(blackbody, sizeof(blackbody)),
          ColorMapper(jet, sizeof(jet)),
          ColorMapper(blue_gold, sizeof(blue_gold)),
          ColorMapper(ice_fire, sizeof(ice_fire)),
          ColorMapper(nic_edge, sizeof(nic_edge)),
          ColorMapper(covise, sizeof(covise)),
          ColorMapper(jamie_draft, sizeof(jamie_draft)),
          ColorMapper(hsv, sizeof(hsv)),
          ColorMapper({vec3f(0,0,0), vec3f(1,1,1)})
      };

      // ==================================================================
      // set up transfer function
      // ==================================================================
      xfColor.resize(colormaps.size());
      for (size_t i=0;i<xfColor.size();i++) {
        xfColor[i].resize(XF_ALPHA_COUNT);
      }
      for (int i=0;i<XF_ALPHA_COUNT;i++) {
        float t=i/(float)(XF_ALPHA_COUNT-1);
        for (size_t j = 0; j < colormaps.size(); ++j) {
            xfColor[j][i] = colormaps[j](t);
        }
      }
      if (cmdline.xfAlpha.empty()) {
        cmdline.xfAlpha.resize(renderer->scalarFields.size());
        for (size_t c=0;c<renderer->scalarFields.size();++c) {
          for (int i=0;i<XF_ALPHA_COUNT;i++) {
            float t=i/(float)(XF_ALPHA_COUNT-1);
            cmdline.xfAlpha[c][i] = t;
          }
        }
      }
      for (size_t c=0;c<renderer->scalarFields.size();++c) {
        xfColorMap[c] = 0;
        xfDomain[c] = renderer->scalarFields[c]->valueRange;
        renderer->updateXF(c,
                           cmdline.xfAlpha[c].data(),
                           xfColor[xfColorMap[c]],
                           xfDomain[c],
                           cmdline.xfOpacityScale);
      }
      xfPanel = new glui::GLUI_Panel(context, "Transfer Function");
      xfWidget = new GLUI_TransferFunction
        ( xfPanel, "Alpha Edit", cmdline.xfAlpha[channel].data(), XF_ALPHA_COUNT,//(int)xfAlpha.size(),
          -1, transferFunctionChangedCB);
      xfWidget->xf_color_array=(float*)xfColor[xfColorMap[channel]].data();
      new glui::GLUI_Column(xfPanel, false);
      new glui::GLUI_Spinner(xfPanel, "Opacity Scale",
                             &cmdline.xfOpacityScale, invalidID, transferFunctionChangedCB);
      if (cmdline.valueRange.lower < cmdline.valueRange.upper) {
        xfDomain[channel] = cmdline.valueRange;
      }

      xfDomainLoSP = new glui::GLUI_Spinner(xfPanel, "xfDomain.lower",
                                            &xfDomain[channel].lower, invalidID, transferFunctionChangedCB);
      xfDomainHiSP = new glui::GLUI_Spinner(xfPanel, "xfDomain.upper",
                                            &xfDomain[channel].upper, invalidID, transferFunctionChangedCB);
      //static glui::GLUI_RadioGroup xfColorMapRG(context,&xfColorMap,invalidID,transferFunctionChangedCB);
      xfColorMapLB = new glui::GLUI_Listbox(xfPanel, "xfColorMap",
                            &xfColorMap[channel], invalidID, transferFunctionChangedCB);
      // Channels
      glui::GLUI_Listbox* xfChannelLB = new glui::GLUI_Listbox(xfPanel, "Channel",
                            &channel, 0, channelChangedCB);
      for (size_t c=0;c<renderer->scalarFields.size();++c) {
        std::string str = renderer->scalarFields[c]->name;
        xfChannelLB->add_item(c,str.c_str());
      }

      for (size_t i = 0; i < colormaps.size(); ++i) {
          xfColorMapLB->add_item(i, colormapNames[i].c_str());
      }

      auto tb = new glui::GLUI_TextBox(xfPanel, customColorMapString, true, invalidID, customColorMapChangedCB);
      new glui::GLUI_Button(xfPanel, "Update custom color map", invalidID, customColorMapSetCB);
      if (cmdline.customColorMapString.empty())
        tb->set_text("0.0,(0,0,0)\n1.0,(1,1,1)");
      else {
        stringReplace(cmdline.customColorMapString,"\\n","\n");
        tb->set_text(cmdline.customColorMapString.c_str());
        customColorMapSetCB(0);
      }
      
      // ==================================================================
      // AO
      // ==================================================================
      //new glui::GLUI_Separator(context);
      new glui::GLUI_Checkbox(context, "ao enabled?",
                              &cmdline.ao.enabled, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(context, "ao length",
                             &cmdline.ao.length, invalidID, renderSettingsChangedCB);

      // ==================================================================
      // time vis
      // ==================================================================
      new glui::GLUI_Separator(context);
      new glui::GLUI_Spinner(context, "clock scale / timevis",
                             &cmdline.clockScale, invalidID, renderSettingsChangedCB);

      // // ==================================================================
      // iso plane(s) editor
      // ==================================================================
      isoPanel = new glui::GLUI_Panel(context, "ISO Surfaces");
      isoPanel->set_alignment(GLUI_ALIGN_LEFT);
      for (int i=0;i<MAX_ISO_SURFACES;i++) {
        new glui::GLUI_Column(isoPanel, false);
        if (cmdline.isovalues.size() > i) {
            isoSurfaceEnabled[i] = 1;
            isoSurfaceValue[i] = cmdline.isovalues[i];
        } else {
            isoSurfaceEnabled[i] = 0;
            isoSurfaceValue[i]
                = xfDomain[isoChannel].lower
                + ((i+1)/(MAX_ISO_SURFACES+1.f))*(xfDomain[isoChannel].upper-xfDomain[isoChannel].lower);
        }
        if (cmdline.isochannels.size() > i) {
          isoSurfaceChannel[i] = cmdline.isochannels[i];
        }
        new glui::GLUI_Spinner(isoPanel, "isoValue",
                               &isoSurfaceValue[i], invalidID, isoSurfacesChangedCB);
        glui::GLUI_Listbox* isoChannelLB =  new glui::GLUI_Listbox(isoPanel, "isoChannel",
                              &isoSurfaceChannel[i], 0, isoSurfacesChangedCB);
        for (size_t c=0;c<renderer->scalarFields.size();++c) {
          std::string str = renderer->scalarFields[c]->name;
          isoChannelLB->add_item(c,str.c_str());
        }
        isoChannelLB->do_selection(isoSurfaceChannel[i]);
        isoChannelLB->set_ptr_val(&isoSurfaceChannel[i]);
        new glui::GLUI_Checkbox(isoPanel, "enabled?",
                                &isoSurfaceEnabled[i], invalidID, isoSurfacesChangedCB);
      }
      renderer->updateIsoValues(isoSurfaceValue,
                                isoSurfaceChannel,
                                isoSurfaceEnabled);

      // // ==================================================================
      // contour plane(s) editor
      // ==================================================================
      contourPanel = new glui::GLUI_Panel(context, "Contour Planes");
      for (int i=0;i<MAX_CONTOUR_PLANES;i++) {
        new glui::GLUI_Column(contourPanel, false);
        if (cmdline.contourplanes.size() > i) {
            contourPlaneEnabled[i] = 1;
            contourPlaneNormal[i] = vec3f(cmdline.contourplanes[i]);
            contourPlaneOffset[i] = cmdline.contourplanes[i].w;
            contourPlaneChannel[i] = 0;
        } else {
            contourPlaneEnabled[i] = 0;
            contourPlaneNormal[i] = vec3f(0.f);
            contourPlaneNormal[i][i%3] = 1.f;
            contourPlaneOffset[i] = .5f;
        }

        if (cmdline.contourchannels.size() > 1)
            contourPlaneChannel[i] = cmdline.contourchannels[i];
        else
            contourPlaneChannel[i] = 0;

        new glui::GLUI_Spinner(contourPanel, "contourPlaneNormal.x",
                               &contourPlaneNormal[i][0], invalidID, contourPlanesChangedCB);
        new glui::GLUI_Spinner(contourPanel, "contourPlaneNormal.y",
                               &contourPlaneNormal[i][1], invalidID, contourPlanesChangedCB);
        new glui::GLUI_Spinner(contourPanel, "contourPlaneNormal.z",
                               &contourPlaneNormal[i][2], invalidID, contourPlanesChangedCB);
        new glui::GLUI_Spinner(contourPanel, "contourPlaneOffset",
                               &contourPlaneOffset[i], invalidID, contourPlanesChangedCB);
        glui::GLUI_Listbox* contourChannelLB =  new glui::GLUI_Listbox(contourPanel, "contourPlaneChannel",
                              &contourPlaneChannel[i], 0, contourPlanesChangedCB);
        for (size_t c=0;c<renderer->scalarFields.size();++c) {
          std::string str = renderer->scalarFields[c]->name;
          contourChannelLB->add_item(c,str.c_str());
        }
        contourChannelLB->do_selection(contourPlaneChannel[i]);
        contourChannelLB->set_ptr_val(&contourPlaneChannel[i]);
        new glui::GLUI_Checkbox(contourPanel, "enabled?",
                                &contourPlaneEnabled[i], invalidID, contourPlanesChangedCB);
      }
      if (!cmdline.contourplanes.empty() || !cmdline.contourchannels.empty()) {
        contourPlanesChangedCB(0);
      }
      
      hPanel = new glui::GLUI_Panel(context, "Horizontal1", GLUI_PANEL_NONE);
      // ==================================================================
      // clip box editor
      // ==================================================================
      clipPanel = new glui::GLUI_Panel(hPanel, "Clip Box");
      clipPanel->set_alignment(GLUI_ALIGN_LEFT);
      new glui::GLUI_Spinner(clipPanel, "lo x",
                             &cmdline.clipBox.coords.lower.x, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(clipPanel, "lo y",
                             &cmdline.clipBox.coords.lower.y, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(clipPanel, "lo z",
                             &cmdline.clipBox.coords.lower.z, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(clipPanel, "hi x",
                             &cmdline.clipBox.coords.upper.x, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(clipPanel, "hi y",
                             &cmdline.clipBox.coords.upper.y, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Spinner(clipPanel, "hi z",
                             &cmdline.clipBox.coords.upper.z, invalidID, renderSettingsChangedCB);
      new glui::GLUI_Checkbox(clipPanel, "clip box on?",
                              &cmdline.clipBox.enabled, invalidID, renderSettingsChangedCB);
      // ==================================================================
      // traces editor
      // ==================================================================
      new glui::GLUI_Column(hPanel, false);
      tracesPanel = new glui::GLUI_Panel(hPanel, "Tracer");
      tracesPanel->set_alignment(GLUI_ALIGN_LEFT);
      glui::GLUI_Listbox* tracesChannel0LB =  new glui::GLUI_Listbox(tracesPanel, "Channel 0",
                            &cmdline.traces.channels.x, 0, contourPlanesChangedCB);
      glui::GLUI_Listbox* tracesChannel1LB =  new glui::GLUI_Listbox(tracesPanel, "Channel 1",
                            &cmdline.traces.channels.y, 0, contourPlanesChangedCB);
      glui::GLUI_Listbox* tracesChannel2LB =  new glui::GLUI_Listbox(tracesPanel, "Channel 2",
                            &cmdline.traces.channels.z, 0, contourPlanesChangedCB);
      for (size_t c=0;c<renderer->scalarFields.size();++c) {
        std::string str = renderer->scalarFields[c]->name;
        tracesChannel0LB->add_item(c,str.c_str());
        tracesChannel1LB->add_item(c,str.c_str());
        tracesChannel2LB->add_item(c,str.c_str());
      }
      tracesChannel0LB->do_selection(cmdline.traces.channels.x);
      tracesChannel1LB->do_selection(cmdline.traces.channels.y);
      tracesChannel2LB->do_selection(cmdline.traces.channels.z);
      new glui::GLUI_Spinner(tracesPanel, "num seeds",
                             &cmdline.traces.numTraces, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "time steps",
                             &cmdline.traces.numTimesteps, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "step size",
                             &cmdline.traces.steplen, invalidID, tracerSettingsChangedCB);
      // 2nd column (seed region)
      new glui::GLUI_Column(tracesPanel, false);
      new glui::GLUI_Spinner(tracesPanel, "seed region lo x",
                             &cmdline.traces.seedRegion.lower.x, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "seed region lo y",
                             &cmdline.traces.seedRegion.lower.y, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "seed region lo z",
                             &cmdline.traces.seedRegion.lower.z, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "seed region hi x",
                             &cmdline.traces.seedRegion.upper.x, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "seed region hi y",
                             &cmdline.traces.seedRegion.upper.y, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Spinner(tracesPanel, "seed region hi z",
                             &cmdline.traces.seedRegion.upper.z, invalidID, tracerSettingsChangedCB);
      new glui::GLUI_Checkbox(tracesPanel, "tracer enabled?",
                              &cmdline.traces.enabled, invalidID, tracerSettingsChangedCB);
      // // ==================================================================
      // // clip plane editor
      // // ==================================================================
      // new glui::GLUI_Separator(context);
      // for (int i=0;i<4;i++)
      //   for (int j=0;j<4;j++)
      //     clipPlaneOrientation[i][j] = (i==j);
      // new GLUI_Rotation(context, "clip plane orientation",
      //                   (float *)&clipPlaneOrientation,invalidID,clipPlaneChangedCB);
      // clipPlaneOrigin = vec3f(0.5f);
      // new glui::GLUI_Spinner(context, "clip plane origin x",
      //                        &clipPlaneOrigin.x, invalidID, clipPlaneChangedCB);
      // new glui::GLUI_Spinner(context, "clip plane origin y",
      //                        &clipPlaneOrigin.y, invalidID, clipPlaneChangedCB);
      // new glui::GLUI_Spinner(context, "clip plane origin z",
      //                        &clipPlaneOrigin.z, invalidID, clipPlaneChangedCB);
      // new glui::GLUI_Checkbox(context, "clip plane on?",
      //                         &clipPlaneEnabled, invalidID, clipPlaneChangedCB);

      // ==================================================================
      new glui::GLUI_Separator(context);
      new glui::GLUI_Checkbox(context, "Enable Space Skipping",
                              &viewer->doSpaceSkipping, invalidID,
                              spaceSkippingChangedCB );
      spaceSkippingChangedCB(0);

      // ==================================================================
      new glui::GLUI_Checkbox(context, "Progressive Refinement",
                              &cmdline.doProgressiveRefinement, invalidID,
                              progressiveRefinementCB );

      new glui::GLUI_Checkbox(context, "Enable Gradient Shading (DVR)",
                              &viewer->gradientShadingDVR, invalidID,
                              gradientShadingDVRChangedCB );

      new glui::GLUI_Checkbox(context, "Enable Gradient Shading (ISO)",
                              &viewer->gradientShadingISO, invalidID,
                              gradientShadingISOChangedCB );

      // ==================================================================
      glui::GLUI_Spinner* spinner = new glui::GLUI_Spinner(context, "Ray Marching Step Size",
                             &cmdline.dt, invalidID, dtChangedCB);
      spinner->set_float_limits(.01f, 100.f);
      // new glui::GLUI_Checkbox(context, "HeatMap (enabled)",
      //                          &heatMapEnabled, invalidID,
      //                          heatMapCB );
      // new glui::GLUI_Spinner(context, "HeatMap (log scale)",
      //                         &heatMapScale, invalidID,
      //                         heatMapCB );
      // new glui::GLUI_Button( context, "Re-center", invalidID, reCenterCB );
      
      // new glui::GLUI_Separator(context);
      // glui::GLUI_Panel *infoPanel = new glui::GLUI_Panel( context, "Data Set Stats" );
      // new glui::GLUI_StaticText(infoPanel, "#bricks = ...\n" );
      // new glui::GLUI_StaticText(infoPanel, "#cells = ..." );

      transferFunctionChangedCB(invalidID);
      renderSettingsChangedCB(invalidID);
      // clipPlaneChangedCB(invalidID);
      // isoSurfacesChangedCB(invalidID);
      int prevChannel = channel;
      for (size_t c=0;c<renderer->scalarFields.size();++c) {
        if (cmdline.colormaps.size() > c) {
          auto fnd = std::find(colormapNames.begin(), colormapNames.end(), cmdline.colormaps[c]);
          if (fnd != colormapNames.end()) {
              xfColorMap[c] = std::distance(colormapNames.begin(), fnd);
              channel = c;
              updateXF();
          } else {
              std::cout << "ERROR: Did not find colormap with name '"
                  << cmdline.colormaps[c] << "', defaulting to Cool Warm\n";
          }
        }
      }
      channel = prevChannel;
    }
    
    // static void heatMapCB(int)
    // {
    //   // renderer.setHeatMapMode(heatMapEnabled,heatMapScale);
    // }

    // static void shadeModeCB(int)
    // {
    //   // renderer.setShadeMode(current->shadeMode);
    // }

    /*! somethign happened that changed an input to the transfer function - re-upload it */
    void updateXF()
    {
      viewer->resetAccumulation();
      xfWidget->xf_color_array=(float*)xfColor[xfColorMap[channel]].data();
      renderer->updateXF(channel,
                         cmdline.xfAlpha[channel].data(),
                         xfColor[xfColorMap[channel]],
                         xfDomain[channel],
                         // interval<float>(min(xfDomainLower,xfDomainUpper),
                         //                 max(xfDomainLower,xfDomainUpper)),
                         // renderer->sf->valueRange,
                         cmdline.xfOpacityScale);
    }
    
    static void transferFunctionChangedCB(int)
    {
      viewer->xfDomain = current->xfDomain;
      current->updateXF();
    }


    void updateChannel()
    {
      viewer->currentChannel = channel;

      xfWidget->xf_alpha=cmdline.xfAlpha[channel].data();
      xfWidget->xf_color_array=(float*)xfColor[xfColorMap[channel]].data();

      //xfDomainLoSP->edittext->set_ptr_val(&xfDomain[channel].lower);
      //xfDomainHiSP->edittext->set_ptr_val(&xfDomain[channel].upper);

      // TODO..
      //xfDomainLoSP->set_float_val(xfDomain[channel].lower);
      //xfDomainHiSP->set_float_val(xfDomain[channel].upper);

      xfColorMapLB->do_selection(xfColorMap[channel]);
      xfColorMapLB->set_ptr_val(&xfColorMap[channel]);
    }

    static void channelChangedCB(int)
    {
      current->updateChannel();
    }

    static void customColorMapChangedCB(GLUI_Control *control)
    {
    }

    static void customColorMapSetCB(int)
    {
      // Print to stdout
      std::string copiedString = current->customColorMapString;
      stringReplace(copiedString,"\n","\\\\n");
      stringReplace(copiedString,"(","\\(");
      stringReplace(copiedString,")","\\)");
      std::cout << "cmdline: --custom-colormap " << copiedString << '\n';

      ColorMapper cm(current->customColorMapString);
      const size_t customColorMapIndex = current->xfColor.size() - 1;
      for (int i=0;i<XF_ALPHA_COUNT;i++) {
        float t=i/(float)(XF_ALPHA_COUNT-1);
        current->xfColor[customColorMapIndex][i] = cm(t);
      }

      viewer->customColorMap = new ColorMapper(current->customColorMapString);
      current->updateXF();
    }

    static void progressiveRefinementCB(int)
    {
      viewer->resetAccumulation();
    }

    static void dtChangedCB(int)
    {
      viewer->resetAccumulation();
    }

    static void renderSettingsChangedCB(int)
    {
      renderer->frameState.clipBox.enabled = cmdline.clipBox.enabled;
      const box3f bounds = renderer->worldSpaceBounds;
      renderer->frameState.clipBox.coords.lower
        = vec3f(bounds.lower) + cmdline.clipBox.coords.lower * vec3f(bounds.span());
      renderer->frameState.clipBox.coords.upper
        = vec3f(bounds.lower) + cmdline.clipBox.coords.upper * vec3f(bounds.span());
      
      renderer->frameState.ao.enabled = cmdline.ao.enabled;
      renderer->frameState.ao.length = cmdline.ao.length;

      renderer->frameState.clockScale = cmdline.clockScale;
      
      viewer->resetAccumulation();
    }

    static void tracerSettingsChangedCB(int)
    {
      renderer->traces.tracerChannels = cmdline.traces.channels;
      renderer->traces.seedRegion = cmdline.traces.seedRegion;
      renderer->traces.numTraces = cmdline.traces.numTraces;
      renderer->traces.numTimesteps = cmdline.traces.numTimesteps;
      renderer->traces.steplen = cmdline.traces.steplen;
      renderer->traces.tracerEnabled = cmdline.traces.enabled;

      renderer->resetTracer();

      viewer->resetAccumulation();
    }
    
    // static void clipPlaneChangedCB(int)
    // {
    //   vec3f clipPlaneNormal(current->clipPlaneOrientation[0][0],
    //                         current->clipPlaneOrientation[0][1],
    //                         current->clipPlaneOrientation[0][2]);
    //   renderer->updateClipPlane(current->clipPlaneOrigin,
    //                             clipPlaneNormal,
    //                             current->clipPlaneEnabled);
    //   viewer->resetAccumulation();
    // }
    
    static void isoSurfacesChangedCB(int)
    {
      viewer->resetAccumulation();
      renderer->updateIsoValues(current->isoSurfaceValue,
                                current->isoSurfaceChannel,
                                current->isoSurfaceEnabled);
    }
    
    static void contourPlanesChangedCB(int)
    {
      viewer->resetAccumulation();
      renderer->updateContourPlanes(current->contourPlaneNormal,
                                    current->contourPlaneOffset,
                                    current->contourPlaneChannel,
                                    current->contourPlaneEnabled);
    }
    
    static void spaceSkippingChangedCB(int)
    {
      viewer->resetAccumulation();
      renderer->setSpaceSkipping(viewer->doSpaceSkipping);
    }

    static void gradientShadingDVRChangedCB(int value)
    {
      viewer->resetAccumulation();
      renderer->setGradientShadingDVR(viewer->gradientShadingDVR);
    }

    static void gradientShadingISOChangedCB(int value)
    {
      viewer->resetAccumulation();
      renderer->setGradientShadingISO(viewer->gradientShadingISO);
    }

    /*! currently active channel */
    int channel=0;
    /*! isos only from channel 0 */
    const int isoChannel=0;
    /*! the widget to control the alpha mapping */
    // std::vector<float> xfAlpha;
    std::vector<std::vector<vec3f>> xfColor;
    /*! ID of he color map to be used for the transfer function (UI
      will flip that int, then we can change the xfColor[] array and
      re-upload) - not yet implemented */
    std::vector<int> xfColorMap;
    /*! @{ the lower/upper range of the scalar range that the trnasfer
        function will cover. Will get initialized to scalar field's
        value range upon gui creation */
    std::vector<interval<float>> xfDomain;
    /*! String edited by the user and that is converted to a custom
        color map on edit */
    GLUI_String customColorMapString;
    /*! @} */

    // float clipPlaneOrientation[4][4];
    // vec3f clipPlaneOrigin;
    // int   clipPlaneEnabled { 0 };
    
    float isoSurfaceValue[MAX_ISO_SURFACES];
    int   isoSurfaceChannel[MAX_ISO_SURFACES];
    int   isoSurfaceEnabled[MAX_ISO_SURFACES];

    vec3f contourPlaneNormal[MAX_CONTOUR_PLANES];
    float contourPlaneOffset[MAX_CONTOUR_PLANES];
    int   contourPlaneChannel[MAX_CONTOUR_PLANES];
    int   contourPlaneEnabled[MAX_CONTOUR_PLANES];

    glui::GLUI_TransferFunction *xfWidget { nullptr };
    glui::GLUI_Listbox *xfColorMapLB { nullptr };
    glui::GLUI_Spinner *xfDomainLoSP { nullptr };
    glui::GLUI_Spinner *xfDomainHiSP { nullptr };
    glui::GLUI_Context *context { nullptr };
    glui::GLUI_Panel* xfPanel { nullptr };
    glui::GLUI_Panel* isoPanel { nullptr };
    glui::GLUI_Panel* contourPanel { nullptr };

    glui::GLUI_Panel* hPanel { nullptr };
    glui::GLUI_Panel* clipPanel { nullptr };
    glui::GLUI_Panel* tracesPanel { nullptr };
    // int doProgressiveRefinement = 1;
    // int shadeMode      = 0;
    // int heatMapEnabled = 0;
    // int heatMapScale   = 0;
    // MyRootWindow   &viewer;
  };
  GLUIInterface *GLUIInterface::current = nullptr;
  

  
  void usage(const std::string &msg)
  {
    if (msg != "")
      std::cout << "Error: " << msg << std::endl << std::endl;
    std::cout << "usage : ./exaViewer path/to/configFile.exa" << std::endl;
    std::cout << "--camera pos.x pos.y pos.z at.x at.y at.z up.x up.y up.z" << std::endl;
    std::cout << "--size windowSize.x windowSize.y" << std::endl;
    std::cout << std::endl;
    exit((msg == "") ? 0 : 1);
  }
  
  extern "C" int main(int argc, char** argv)
  {
    try {
      Config::SP config;
      for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg[0] != '-') {
          config = Config::parseConfigFile(arg);
        } else if (arg == "-win" || arg == "--size") {
          cmdline.windowSize.x = std::atoi(argv[++i]);
          cmdline.windowSize.y = std::atoi(argv[++i]);
        } else if (arg == "--display-string") {
          cmdline.displayString = argv[++i];
        } else if (arg == "--gradientShadingDVR") {
          std::string flag = argv[++i];
          if (flag == "on" || flag == "1")
            cmdline.gradientShadingDVR = true;
          else if (flag == "off" || flag == "0") 
            cmdline.gradientShadingDVR = false;
          else
            throw std::runtime_error("invalid value '"+flag+"' for --gradientShadingDVR argument");
        } else if (arg == "--gradientShadingISO") {
          std::string flag = argv[++i];
          if (flag == "on" || flag == "1")
            cmdline.gradientShadingISO = true;
          else if (flag == "off" || flag == "0") 
            cmdline.gradientShadingISO = false;
          else
            throw std::runtime_error("invalid value '"+flag+"' for --gradientShadingISO argument");
        } else if (arg == "--colorbar") {
          std::string flag = argv[++i];
          if (flag == "on" || flag == "1")
            cmdline.showColorBar = true;
          else if (flag == "off" || flag == "0") 
            cmdline.showColorBar = false;
          else
            throw std::runtime_error("invalid value '"+flag+"' for --colorbar argument");
        } else if (arg == "--colorbar-channel") {
          cmdline.colorBarChannel = std::atoi(argv[++i]);
          cmdline.showColorBar = true;
        } else if (arg == "--ao") {
          std::string flag = argv[++i];
          if (flag == "on" || flag == "1")
            cmdline.ao.enabled = true;
          else if (flag == "off" || flag == "0") 
            cmdline.ao.enabled = false;
          else
            throw std::runtime_error("invalid value '"+flag+"' for --ao argument");
        } else if (arg == "--ao-length") {
          cmdline.ao.length = std::stof(argv[++i]);
        } else if (arg == "--no-pg") {
          cmdline.doProgressiveRefinement = false;
        // } else if (arg == "--no-pg") {
        //   std::string flag = argv[++i];
        //   if (flag == "on" || flag == "1")
        //     cmdline.doProgressiveRefinement = true;
        //   else if (flag == "off" || flag == "0") 
        //     cmdline.doProgressiveRefinement = false;
        //   else
        //     throw std::runtime_error("invalid value '"+flag+"' for --ao argument");
        } else if (arg == "--xf") {
          const std::string fileName = argv[++i];
          std::cout << "#viewer: reading transfer function from '" << fileName << "'" << std::endl;
          std::ifstream xfFile(fileName,std::ios::binary);
          cmdline.xfAlpha.resize(cmdline.xfAlpha.size()+1);
          xfFile.read((char*)cmdline.xfAlpha.back().data(),XF_ALPHA_COUNT*sizeof(float));
        } else if (arg == "--camera") {
          cmdline.camera.vp.x = std::atof(argv[++i]);
          cmdline.camera.vp.y = std::atof(argv[++i]);
          cmdline.camera.vp.z = std::atof(argv[++i]);
          cmdline.camera.vi.x = std::atof(argv[++i]);
          cmdline.camera.vi.y = std::atof(argv[++i]);
          cmdline.camera.vi.z = std::atof(argv[++i]);
          cmdline.camera.vu.x = std::atof(argv[++i]);
          cmdline.camera.vu.y = std::atof(argv[++i]);
          cmdline.camera.vu.z = std::atof(argv[++i]);
        } else if (arg == "--fov") {
            cmdline.camera.fov = std::atof(argv[++i]);
        } else if (arg == "--range") {
            cmdline.valueRange.lower = std::atof(argv[++i]);
            cmdline.valueRange.upper = std::atof(argv[++i]);
        } else if (arg == "--colormap") {
            cmdline.colormaps.resize(cmdline.colormaps.size()+1);
            cmdline.colormaps.back() = argv[++i];
        } else if (arg == "--custom-colormap") {
            cmdline.customColorMapString = argv[++i];std::cout << cmdline.customColorMapString << '\n';
        } else if (arg == "--xf-scale") {
            cmdline.xfOpacityScale = std::atof(argv[++i]);
        } else if (arg == "--isovals") {
            for (size_t j = i + 1; j < argc; ++j) {
                if (argv[j][0] == '-') {
                    break;
                }
                if (cmdline.isovalues.size() == MAX_ISO_SURFACES) {
                    std::cout << "Warning: too many isovalues specified, max is: "
                        << MAX_ISO_SURFACES << "\n";
                }
                cmdline.isovalues.push_back(std::atof(argv[j]));
                ++i;
            }
        } else if (arg == "--isochans") {
            for (size_t j = i + 1; j < argc; ++j) {
                if (argv[j][0] == '-') {
                    break;
                }
                if (cmdline.isochannels.size() == MAX_ISO_SURFACES) {
                    std::cout << "Warning: too many iso channels specified, max is: "
                        << MAX_ISO_SURFACES << "\n";
                }
                cmdline.isochannels.push_back(std::atoi(argv[j]));
                ++i;
            }
        } else if (arg == "--contourplane") {
          vec4f plane;
          for (int j=i+1; j<i+1+4; ++j) {
            plane[j-i-1] = std::atof(argv[j]);
          }
          i += 4;
          cmdline.contourplanes.push_back(plane);
        } else if (arg == "--contourchan") {
          cmdline.contourchannels.push_back(std::atoi(argv[++i]));
        } else if (arg == "--clip-box") {
            cmdline.clipBox.coords.lower.x = std::atof(argv[++i]);
            cmdline.clipBox.coords.lower.y = std::atof(argv[++i]);
            cmdline.clipBox.coords.lower.z = std::atof(argv[++i]);
            cmdline.clipBox.coords.upper.x = std::atof(argv[++i]);
            cmdline.clipBox.coords.upper.y = std::atof(argv[++i]);
            cmdline.clipBox.coords.upper.z = std::atof(argv[++i]);
            cmdline.clipBox.enabled = 1;
        } else if (arg == "--dt") {
            cmdline.dt = std::atof(argv[++i]);
        } else {
          throw std::runtime_error("unrecognized parameter '" + arg + "'");
        }
      }
      if (!config)
        usage("no config file specified");
      
      if (!config->bricks.sp)
        usage("no bricks file specified");

      std::string inFileName;

      // if (argc != 4 && argc != 3) throw std::runtime_error("exaViewer <cellFile> <scalarFile>");
      ExaBricks::SP input = config->bricks.sp;//ExaBricks::load(argv[1],{argv[2]});
      const box3f bounds = config->getBounds();

      std::vector<TriangleMesh::SP> surfaces = config->surfaces;
      size_t numVerts = 0;
      size_t numIndices = 0;
      for (const auto &s : surfaces) {
          numVerts += s->vertex.size();
          numIndices += s->index.size();
      }
      if (numIndices != 0) {
          std::cout << "Loaded mesh consisting of "
                    << owl::prettyDouble(numIndices) << " triangles\n"
                    << "Vertices Memory: "
                    << owl::prettyNumber(numIndices * sizeof(vec3i)) << " bytes\n"
                    << "Indices Memory: "
                    << owl::prettyNumber(numIndices * sizeof(vec3f)) << " bytes\n";
      }

      // if (argc == 4)
      //   surface = TriangleMesh::load(argv[3]);

      /* ==================================================================
         create the optix renderer
         ================================================================== */
      renderer
        = std::make_shared<OptixRenderer>(config->bricks.sp,
                                          config->surfaces,
                                          config->scalarFields);
      renderer->setVoxelSpaceTransform(config->bricks.voxelSpaceTransform);

      /* ==================================================================
         create the root window that controls the renderer and gui
         ================================================================== */
      viewer = std::make_shared<MyRootWindow>(argc, argv, bounds);

      /* ==================================================================
         build the actual widget(s) ....
         ================================================================== */
      // imglui::UI::SP glui = imglui::UI::create("imGlui TransferFunction Example");
      // glui->add(imglui::TextWidget::create("The following is a array of opacity values"));
      // window->xfWidget = imglui::TransferFunctionWidget::create(XF_ALPHA_COUNT);
      // glui->add(window->xfWidget);
      // window->xfWidget->setValueRange(scalarField->valueRange.lower,
      //                                 scalarField->valueRange.upper);
      // window->xfWidget->setHistogram(computeHistogram(scalarField));
      // glui->add(imglui::TextWidget::create("Edit by left-clicking and 'drawing' with the mouse"));
      // window->ui = glui;

      viewer->enableFlyMode();
      viewer->enableInspectMode(bounds);
      viewer->setWorldScale(owl::length(bounds.span()));
      if (cmdline.camera.vu != vec3f(0.f)) {
        viewer->setCameraOrientation(/*origin   */cmdline.camera.vp,
                                           /*lookat   */cmdline.camera.vi,
                                           /*up-vector*/cmdline.camera.vu,
                                           /*fovy(deg)*/cmdline.camera.fov);
      } else {
        viewer->setCameraOrientation(/*origin   */
                                           bounds.center()
                                           + vec3f(-.3f, .7f, +1.f) * bounds.span(),
                                           /*lookat   */bounds.center(),
                                           /*up-vector*/vec3f(0.f, 1.f, 0.f),
                                           /*fovy(deg)*/70.f);
      }
      if (cmdline.windowSize != vec2i(0)) {
        glutReshapeWindow(cmdline.windowSize.x,cmdline.windowSize.y);//glutWindow->glutWindowHandle,
      }


      // ------------------------------------------------------------------
      // create user interface
      // ------------------------------------------------------------------
      GLUIInterface ui(viewer->getNativeHandle());

      /* ==================================================================
         and run the whole shebang ...
         ================================================================== */
      viewer->showAndRun();
    }
    catch (std::runtime_error e) {
      std::cerr << "Fatal error " << e.what() << std::endl;
      exit(1);
    }
    return 0;
  }
}
