#include "Scene.h"

#include "ColladaLoader.h"
#include "raytracer.h"

#include <GL/glew.h>
#include <GL/glut.h>


#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include <iostream>
#include <fstream>
using namespace std;

CScene CScene::m_instance;

bool dummyTGAHeader(std::ifstream &currentfile, int &sizeX, int &sizeY)
{
	char dummy;
	char temp;
	currentfile.get(dummy).get(dummy);
	currentfile.get(temp);                  
	if (temp!=2)
		return false;
	currentfile.get(dummy).get(dummy);
	currentfile.get(dummy).get(dummy);
	currentfile.get(dummy);
	currentfile.get(dummy).get(dummy);           
	currentfile.get(dummy).get(dummy);           
	currentfile.get(temp);
	sizeX = temp;
	currentfile.get(temp);
	sizeX += temp * 256;

	currentfile.get(temp);
	sizeY = temp;
	currentfile.get(temp);
	sizeY += temp * 256;

    currentfile.get(temp);                 
	currentfile.get(dummy);
	return true;
}

bool cubemap::Init(char * name[6])
{
	ifstream currentfile ;
	color * currentColor;
	int x,y, dummySizeX, dummySizeY;

    currentfile.open(name[up], ios_base::binary);

	if ((!currentfile)||(!dummyTGAHeader(currentfile, sizeX, sizeY)))
		return false;
    if (sizeX <= 0 || sizeY <= 0)
        return false;

    color * textureCPU = new color[sizeX * sizeY * 6];

	int number = cubemap::up * sizeX * sizeY;
	currentColor = 	textureCPU + number;
	for (y = 0; y < sizeY; y++)
	for (x = 0; x < sizeX; x++)
	{
		currentColor->blue = currentfile.get() /255.0f;
		currentColor->green = currentfile.get() /255.0f;
		currentColor->red = currentfile.get() /255.0f;
		currentColor++;
	}
	currentfile.close();
	
    for (unsigned i = cubemap::down; i <= cubemap::backward; ++i)
    {
	    number = i * sizeX * sizeY;
	    currentColor = 	textureCPU + number;
	    currentfile.open(name[i], ios_base::binary);
	    if ((!currentfile)||
            (!dummyTGAHeader(currentfile, dummySizeX, dummySizeY)) ||
            sizeX != dummySizeX || 
            sizeY != dummySizeY)
        {
            // Les textures doivent être de taille identique..
			free(textureCPU);
		    return false;
        }

	    for (y = 0; y < sizeY; y++)
	    for (x = 0; x < sizeX; x++)
	    {
			currentColor->blue = currentfile.get() /255.0f;
			currentColor->green = currentfile.get() /255.0f;
			currentColor->red = currentfile.get() /255.0f;
			currentColor++;
	    }
	    currentfile.close();
    }


	cudaMalloc((void**) &textureCUDA, sizeof(color) * sizeX * sizeY * 6);
	cudaMemcpy(textureCUDA,textureCPU,sizeof(color)* sizeX * sizeY * 6,cudaMemcpyHostToDevice);

	delete (textureCPU);

	return true;
}




void CScene::debugMode()
{

	ambientLight = 0.3f;

	light l1;
	light l2;
	light l3;
	light l4;
	light l5;
	light l6;
	light l7;
	light l8;

	l1.pos.x = 1000;l1.pos.y = 1000;l1.pos.z = 1000;
	l2.pos.x = -200;l2.pos.y = 200;l2.pos.z = -200;
	l3.pos.x = -1000;l3.pos.y = -1000;l3.pos.z = -1000;
	l4.pos.x = 1000;l4.pos.y = -1000;l4.pos.z =  1000;
	l5.pos.x = 1000;l5.pos.y = 1000;l5.pos.z = -1000;
	l6.pos.x = 1000;l6.pos.y = -1000;l6.pos.z = -1000;
	l7.pos.x = -1000;l7.pos.y = 1000;l7.pos.z = 1000;
	l8.pos.x = -1000;l8.pos.y = -1000;l8.pos.z =  1000;

	l1.intensity.blue =1.0; l1.intensity.red = 1.0; l1.intensity.green = 1.0;
	l2.intensity.blue = 1.0; l2.intensity.red = 1.0; l2.intensity.green = 1.0;
	l3.intensity.blue = 1.0; l3.intensity.red = 1.0; l3.intensity.green = 1.0;
	l4.intensity.blue = 1.0; l4.intensity.red = 1.0; l4.intensity.green = 1.0;
	l5.intensity.blue =1.0; l5.intensity.red = 1.0; l5.intensity.green = 1.0;
	l6.intensity.blue = 1.0; l6.intensity.red = 1.0; l6.intensity.green = 1.0;
	l7.intensity.blue = 1.0; l7.intensity.red = 1.0; l7.intensity.green = 1.0;
	l8.intensity.blue = 1.0; l8.intensity.red = 1.0; l8.intensity.green = 1.0;

	lightContainer.push_back(l1);
	lightContainer.push_back(l2);
	//lightContainer.push_back(l3);
	lightContainer.push_back(l4);
	//lightContainer.push_back(l5);
	lightContainer.push_back(l6);
	//lightContainer.push_back(l7);
	//lightContainer.push_back(l8);

	sphere* S1 = NULL;

	for(int i =0; i<5; i++)
	{
		for(int j = 0; j < 5; j++)
		{
			for(int k=0; k <5; k++)
			{
				S1 = new sphere();
				S1->materialId = materialContainer.size() + ((i+j+k)%3);
				S1->pos.x = -200 + i*100;
				S1->pos.y = -200 + j*100;
				S1->pos.z = -200 + k*100;
				S1->size = 20;
					
				sphereContainer.push_back(*S1);
			}
		}
	}


	material  m1;

	m1.ambient.blue = 0.56;  m1.ambient.red = 0.14; m1.ambient.green = 0.42;

	m1.diffuse.blue = 0.56;  m1.diffuse.red = 0.14;  m1.diffuse.green = 0.42;

	m1.specular.blue = 1.0; m1.specular.red = 1.0; m1.specular.green = 1.0;

	m1.reflectivity = 0.3;

	m1.transparency = 0.0;

	m1.shininess = 20.0;


	material  m2;

	m2.ambient.blue = 1.0;  m2.ambient.red =1.0; m2.ambient.green =1.0;

	m2.diffuse.blue = 1.0;  m2.diffuse.red = 1.0;  m2.diffuse.green = 1.0;

	m2.specular.blue = 1.0; m2.specular.red = 1.0; m2.specular.green = 1.0;

	m2.reflectivity = 0.05;

	m2.transparency = 0.95;

	m2.shininess = 2.0;

	material  m3;

	m3.ambient.blue = 0.5;  m3.ambient.red = 0.5; m3.ambient.green = 0.5;

	m3.diffuse.blue = 0.5;  m3.diffuse.red = 0.5;  m3.diffuse.green = 0.5;

	m3.specular.blue = 0.5; m3.specular.red = 0.5; m3.specular.green = 0.5;

	m3.reflectivity = 0.80;

	m3.transparency = 0.50;

	m3.shininess = 60.0;

	materialContainer.push_back(m1);
	materialContainer.push_back(m2);
	materialContainer.push_back(m3);

	//Devra être initialisé dans le fichier!
	mCamera.position.x = 0; 
	mCamera.position.y = 0;
	mCamera.position.z = -500;

	mCamera.focale = 500;
	mCamera.znear = 0.1f;
	mCamera.zfar = 100.0f;

	mCamera.vXAxis.x = 1;
	mCamera.vXAxis.y = 0;
	mCamera.vXAxis.z = 0;

	mCamera.vYAxis.x = 0;
	mCamera.vYAxis.y = 1;
	mCamera.vYAxis.z = 0;

	mCamera.vZAxis = mCamera.vXAxis^mCamera.vYAxis;

	printf("DEBUG PARAMETERS LOADED.\n");
}


bool CScene::initScene()
{
	/*MATERIAL DEFAUT*/

	material  m1;

	m1.ambient.blue = 1.0;  m1.ambient.red = 1.0; m1.ambient.green = 1.00;

	m1.diffuse.blue = 1.0;  m1.diffuse.red = 1.00;  m1.diffuse.green = 1.00;

	m1.specular.blue = 1.0; m1.specular.red = 1.00; m1.specular.green = 1.00;

	m1.reflectivity = 1.0;

	m1.transparency = 0.0;

	m1.shininess = 10.0;

	materialContainer.push_back(m1);

	if (!CColladaLoader::getSingletonPtr()->loadCollada("scene.dae",*this))
	{
		printf("Failure when loading the collada file.\n Push key to continue.\n");
		getchar();
		return false;
	}

	debugMode();	

	printf("DEBUG PARAMETERS LOADING.\n");

	char * name[6];
	
	name[cubemap::up] = (char * ) malloc(sizeof(char) * strlen ("violentdays_up.tga"));
	strcpy(name[cubemap::up] , "violentdays_up.tga");
	name[cubemap::down] = (char * ) malloc(sizeof(char) * strlen ("violentdays_dn.tga"));
    strcpy(name[cubemap::down] , "violentdays_dn.tga");
	name[cubemap::right] = (char * ) malloc(sizeof(char) * strlen ("violentdays_rt.tga"));
    strcpy(name[cubemap::right] , "violentdays_ft.tga");
	name[cubemap::left] = (char * ) malloc(sizeof(char) * strlen ("violentdays_lf.tga"));
    strcpy(name[cubemap::left] , "violentdays_bk.tga");
	name[cubemap::forward] = (char * ) malloc(sizeof(char) * strlen ("violentdays_ft.tga"));
    strcpy(name[cubemap::forward] ,  "violentdays_rt.tga");
	name[cubemap::backward] = (char * ) malloc(sizeof(char) * strlen ("violentdays_bk.tga"));
	strcpy(name[cubemap::backward] ,  "violentdays_lf.tga");

    mCubeMap.bExposed = false;
    mCubeMap.bsRGB = true;
    mCubeMap.exposure = 1.0;


	mCubeMap.Init(name);


	pRootBSP = build_BSP(meshes,&pMemCuda);

	return true;
}

