#ifndef _CUDADEVICE_H
#define _CUDADEVICE_H

#include <GL/glew.h>
#include <GL/glut.h>


#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "scene.h"
#include "BSP.h"


__device__ color readTexture(const color* tab, float u, float v, int sizeU, int sizeV);

__device__ color readCubemap(const cubemap &cm, ray myray);

__device__ bool HitBoundingBox(point min,point max,point orig,vecteur vdir,point & pCoord);

__global__ void
cudaProcess(float* g_odata, point		positionCam,
							vecteur vXAxis,	vecteur vYAxis,vecteur vZAxis,
							material	vectorMat[], int nbMat,
						    sphere    vectorSphere[], int nbSphere, 
						    light	vectorLight[], int nbLights,
							cubemap cm,
							nodeCuda * rootNode,triangle * pMemTriangles);

__device__ void normalise(vecteur* dir);

__device__ bool hitSphere(const ray &r, const sphere &s, float &t);

__device__ bool intersect_RayTriangle(const ray &R, const triangle& T, float & distance, bool shadowCast);

__device__ void cudaAddRay(ray viewRay, 
							material	vectorMat[],
							int nbMat,
						    sphere    vectorSphere[],
							int nbSphere, 
						    light	vectorLight[],
							int nbLights,
							cubemap cm,
							nodeCuda* root,triangle * pMemTriangles,
						    color*	output);

__device__ void cudaCalcTransparencyRay(ray RefractedRay, 
							material	vectorMat[],
							int nbMat,
						    sphere    vectorSphere[],
							int nbSphere, 
						    light	vectorLight[],
							int nbLights,
							cubemap cm,
							nodeCuda* root,triangle * pMemTriangles,
							//float coefReflect,float coefRefract,
						    color*	output);

__device__ void cudaCalcTransparencyRay2(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
							//float coefReflect,float coefRefract,
						    color*	output);

__device__ void cudaCalcTransparencyRay3(ray RefractedRay, 
							material	vectorMat[],
							int nbMat,
						    sphere    vectorSphere[],
							int nbSphere, 
						    light	vectorLight[],
							int nbLights,
							cubemap cm,
							nodeCuda* root,triangle * pMemTriangles,
							//float coefReflect,float coefRefract,
						    color*	output);

__device__ void cudaCalcTransparencyRay4(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
							//float coefReflect,float coefRefract,
						    color*	output);


#endif