#ifndef __RAYTRACE__
#define __RAYTRACE__

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
//#include <rendercheck_gl.h>

#include "scene.h"
#include "cudaDevice.h"
#include "BSP.h"

#include <list>



void
launch_cudaProcess(dim3 grid, dim3 block, float* g_odata, 
									int imgw, int imgh,
									point positionCamera,
									vecteur vXAxis,	vecteur vYAxis,vecteur vZAxis,
									std::vector<material>, 
									std::vector<sphere>, 
									std::vector<light>,
									cubemap cm,
									nodeCuda * root,triangle * memoryCuda);

nodeCuda * build_BSP(std::list<mesh> meshes,triangle ** memoryCuda);


#endif