#ifndef __DISPLAY_H__
#define __DISPLAY_H__


#include <windows.h>

// includes, GL

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
//#include <rendercheck_gl.h>

// includes
#include "rayTracer.h"

void eventSouris(int button, int state, int x, int y);

void eventSourisMouv(int x, int y);

void eventClavier(unsigned char key ,int x,int y);

bool initGL(int argc, char **argv );

bool initGLBuffers();

void process();

void display();

#endif