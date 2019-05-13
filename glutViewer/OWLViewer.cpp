// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include <GL/glut.h>

#include "OWLViewer.h"
#include "Camera.h"
#include "InspectMode.h"
#include "FlyMode.h"

// eventually to go into 'apps/'
#include "submodules/3rdParty/stb_image_write.h"

namespace owl {
  namespace glutViewer {

    OWLViewer* self = nullptr;

    int OWLViewer::handle = 0;

    void initGLUT(int argc, char** argv)
    {
      static bool alreadyInitialized = false;
      if (alreadyInitialized) return;
      glutInit(&argc,argv);
      std::cout << "#owl.glutViewer: glut initialized" << std::endl;
      alreadyInitialized = true;
    }
    
    /*! helper function that dumps the current frame buffer in a png
        file of given name */
    void OWLViewer::screenShot(const std::string &fileName)
    {
      const uint32_t *fb
        = (const uint32_t*)fbPointer;
      
      std::vector<uint32_t> pixels;
      for (int y=0;y<fbSize.y;y++) {
        const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
        for (int x=0;x<fbSize.x;x++) {
          pixels.push_back(line[x] | (0xff << 24));
        }
      }
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     pixels.data(),fbSize.x*sizeof(uint32_t));
      std::cout << "#owl.glutViewer: frame buffer written to " << fileName << std::endl;
    }
      
    vec2i OWLViewer::getScreenSize()
    {
      int w = glutGet(GLUT_SCREEN_WIDTH);
      int h = glutGet(GLUT_SCREEN_HEIGHT);
      return { w, h };
    }
 
    inline float computeStableEpsilon(float f)
    {
      return abs(f) * float(1./(1<<21));
    }

    inline float computeStableEpsilon(const vec3f v)
    {
      return max(max(computeStableEpsilon(v.x),
                     computeStableEpsilon(v.y)),
                 computeStableEpsilon(v.z));
    }
    
    SimpleCamera::SimpleCamera(const Camera &camera)
    {
      auto &easy = *this;
      easy.lens.center = camera.position;
      easy.lens.radius = 0.f;
      easy.lens.du     = camera.frame.vx;
      easy.lens.dv     = camera.frame.vy;

      const float minFocalDistance
        = max(computeStableEpsilon(camera.position),
              computeStableEpsilon(camera.frame.vx));

      /*
        tan(fov/2) = (height/2) / dist
        -> height = 2*tan(fov/2)*dist
      */
      float screen_height
        = 2.f*tanf(camera.fovyInDegrees/2.f * (float)M_PI/180.f)
        * max(minFocalDistance,camera.focalDistance);
      easy.screen.vertical   = screen_height * camera.frame.vy;
      easy.screen.horizontal = screen_height * camera.aspect * camera.frame.vx;
      easy.screen.lower_left
        = //easy.lens.center
        /* NEGATIVE z axis! */
        - max(minFocalDistance,camera.focalDistance) * camera.frame.vz
        - 0.5f * easy.screen.vertical
        - 0.5f * easy.screen.horizontal;
      // easy.lastModified = getCurrentTime();
    }
    
    // ==================================================================
    // actual viewerwidget class
    // ==================================================================

    void OWLViewer::resize(const vec2i &newSize)
    {
      if (fbPointer)
        cudaFree(fbPointer);
      cudaMallocManaged(&fbPointer,newSize.x*newSize.y*sizeof(uint32_t));
      
      fbSize = newSize;
      if (fbTexture == 0) {
        glGenTextures(1, &fbTexture);
      } else {
        cudaGraphicsUnregisterResource(cuDisplayTexture);
      }
      
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, newSize.x, newSize.y, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, nullptr);
      
      // We need to re-register when resizing the texture
      cudaGraphicsGLRegisterImage(&cuDisplayTexture, fbTexture, GL_TEXTURE_2D, 0);
    }

    
    /*! re-draw the current frame. This function itself isn't
      virtual, but it calls the framebuffer's render(), which
      is */
    void OWLViewer::draw()
    {
      glPushAttrib(GL_ALL_ATTRIB_BITS);

      cudaGraphicsMapResources(1, &cuDisplayTexture);

      cudaArray_t array;
      cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0);
      {
        // sample.copyGPUPixels(cuDisplayTexture);
        cudaMemcpy2DToArray(array,
                            0,
                            0,
                            reinterpret_cast<const void *>(fbPointer),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.y,
                            cudaMemcpyDeviceToDevice);
      }
      cudaGraphicsUnmapResources(1, &cuDisplayTexture);
      
      glDisable(GL_LIGHTING);
      glColor3f(1, 1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, fbTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      
      glDisable(GL_DEPTH_TEST);

      glViewport(0, 0, fbSize.x, fbSize.y);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
      
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
      
        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)fbSize.x, 0.f, 0.f);
      }
      glEnd();

      glPopAttrib();
    }

    /*! re-computes the 'camera' from the 'cameracontrol', and notify
      app that the camera got changed */
    void OWLViewer::updateCamera()
    {
      // camera.digestInto(simpleCamera);
      // if (isActive)
      camera.lastModified = getCurrentTime();
    }

    void OWLViewer::enableInspectMode(RotateMode rm,
                                      const box3f &validPoiRange,
                                      float minPoiDist,
                                      float maxPoiDist)
    {
      inspectModeManipulator
        = std::make_shared<CameraInspectMode>
        (this,validPoiRange,minPoiDist,maxPoiDist,
         rm==POI? CameraInspectMode::POI: CameraInspectMode::Arcball);
      // if (!cameraManipulator)
      cameraManipulator = inspectModeManipulator;
    }

    void OWLViewer::enableInspectMode(const box3f &validPoiRange,
                                      float minPoiDist,
                                      float maxPoiDist)
    {
      enableInspectMode(POI,validPoiRange,minPoiDist,maxPoiDist);
    }

    void OWLViewer::enableFlyMode()
    {
      flyModeManipulator
        = std::make_shared<CameraFlyMode>(this);
      // if (!cameraManipulator)
      cameraManipulator = flyModeManipulator;
    }

    /*! this gets called when the window determines that the mouse got
      _moved_ to the given position */
    void OWLViewer::mouseMotion(const vec2i &newMousePosition)
    {
      if (lastMousePosition != vec2i(-1)) {
        if (leftButton.isPressed)
          mouseDragLeft  (newMousePosition,newMousePosition-lastMousePosition);
        if (centerButton.isPressed)
          mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
        if (rightButton.isPressed)
          mouseDragRight (newMousePosition,newMousePosition-lastMousePosition);
      }
      lastMousePosition = newMousePosition;
    }

    void OWLViewer::mouseDragLeft  (const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragLeft(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void OWLViewer::mouseDragCenter(const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragCenter(where,delta);
    }

    /*! mouse got dragged with left button pressedn, by 'delta' pixels, at last position where */
    void OWLViewer::mouseDragRight (const vec2i &where, const vec2i &delta)
    {
      if (cameraManipulator) cameraManipulator->mouseDragRight(where,delta);
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonLeft  (const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonLeft(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonCenter(const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonCenter(where,pressed);

      lastMousePosition = where;
    }

    /*! mouse button got either pressed or released at given location */
    void OWLViewer::mouseButtonRight (const vec2i &where, bool pressed)
    {
      if (cameraManipulator) cameraManipulator->mouseButtonRight(where,pressed);

      lastMousePosition = where;
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void OWLViewer::key(char key, const vec2i &where)
    {
      if (cameraManipulator) cameraManipulator->key(key,where);
    }

    /*! this gets called when the user presses a key on the keyboard ... */
    void OWLViewer::special(int key, int mods, const vec2i &where)
    {
      if (cameraManipulator) cameraManipulator->special(key,where);
    }


    void OWLViewer::setTitle(const std::string &s)
    {
      glutSetWindowTitle(s.c_str());
    }
    
    OWLViewer::OWLViewer(int argc,
                         char** argv,
                         const std::string &title,
                         const vec2i &initWindowSize)
    {
      self = this;

      initGLUT(argc, argv);
    
      // that should give us a GL 2.1 context or so
      glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

      glutInitWindowSize(initWindowSize.x, initWindowSize.y);
      handle = glutCreateWindow(title.c_str());

    }


    /*! callback for a window resizing event */
    static void glut_reshape_cb(int width, int height )
    {
      self->resize(vec2i(width,height));
    }

    /*! callback for keyboard input */
    static void glut_key_cb(unsigned char key, int, int)
    {
      self->key((int)key,self->getMousePos());
    }

    /*! callback for special keyboard input */
    static void glut_special_cb(int key, int, int)
    {
      self->special(key,glutGetModifiers(),self->getMousePos());
    }

    /*! callback for _moving_ the mouse to a new position */
    static void glut_motion_cb(int x, int y) 
    {
      self->mouseMotion(vec2i(x,y));
    }

    static void glut_display_cb()
    {
      static double lastCameraUpdate = -1.f;
      if (self->camera.lastModified != lastCameraUpdate) {
        self->cameraChanged();
        lastCameraUpdate = self->camera.lastModified;
      }
      self->render();
      self->draw();
     
      glutSwapBuffers();

      glutPostRedisplay();
    }

    static void glut_idle_cb()
    {
      glutPostRedisplay();
    }

    /*! callback for pressing _or_ releasing a mouse button*/
    static void glut_mouseButton_cb(int button, int state, int /*x*/, int /*y*/) 
    {
      self->mouseButton(button,state,glutGetModifiers());
    }
  
    void OWLViewer::mouseButton(int button, int state, int mods) 
    {
      const bool pressed = (state == GLUT_DOWN);
      lastMousePos = getMousePos();
      switch(button) {
      case GLUT_LEFT_BUTTON:
        leftButton.isPressed        = pressed;
        leftButton.shiftWhenPressed = (mods & GLUT_ACTIVE_SHIFT  );
        leftButton.ctrlWhenPressed  = (mods & GLUT_ACTIVE_CTRL);
        leftButton.altWhenPressed   = (mods & GLUT_ACTIVE_ALT    );
        mouseButtonLeft(lastMousePos, pressed);
        break;
      case GLUT_MIDDLE_BUTTON:
        centerButton.isPressed = pressed;
        centerButton.shiftWhenPressed = (mods & GLUT_ACTIVE_SHIFT  );
        centerButton.ctrlWhenPressed  = (mods & GLUT_ACTIVE_CTRL);
        centerButton.altWhenPressed   = (mods & GLUT_ACTIVE_ALT    );
        mouseButtonCenter(lastMousePos, pressed);
        break;
      case GLUT_RIGHT_BUTTON:
        rightButton.isPressed = pressed;
        rightButton.shiftWhenPressed = (mods & GLUT_ACTIVE_SHIFT  );
        rightButton.ctrlWhenPressed  = (mods & GLUT_ACTIVE_CTRL);
        rightButton.altWhenPressed   = (mods & GLUT_ACTIVE_ALT    );
        mouseButtonRight(lastMousePos, pressed);
        break;
      }
    }


    void OWLViewer::showAndRun()
    {
      int width, height;
      // PRINT(vec2i(width,height));

      glutReshapeFunc(glut_reshape_cb);
      glutMouseFunc(glut_mouseButton_cb);
      glutKeyboardFunc(glut_key_cb);
      glutPassiveMotionFunc(glut_motion_cb);
      glutMotionFunc(glut_motion_cb);
      glutDisplayFunc(glut_display_cb);
      glutIdleFunc(glut_idle_cb);
    
      glutMainLoop();
    }


    vec2i OWLViewer::getMousePos() const
    {
      return lastMousePos;
    }


    int OWLViewer::getNativeHandle() const
    {
      return self->handle;
    }

  } // ::owl::glutViewer
} // ::owl
