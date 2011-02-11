#ifndef __SCENE_H
#define __SCENE_H

#include <Vector>
#include <list>
#include <string>

#include "Def.h"
#include "Mesh.h"
#include "Camera.h"
#include "Matrix.h"
#include "BSP.h"

struct meshNode
{
	matrice44 Quaternion;	//Rotation de l'objet et scale
	vecteur	 position;		//centre du mesh
	std::string name;		//Nom de l'objet pour le retrouver
	std::string material;	//Nom du materiel pour le retrouver
};

struct matRef
{
	unsigned int id;
	std::string material;	//Nom du materiel pour le retrouver
	std::string materialFX;	//Nom du materiel pour le retrouver
};


class CScene
{
	static CScene m_instance;

public :

	static CScene * getSingletonPtr(){return &m_instance;}

	CScene(){pRootBSP = NULL;}
	~CScene(){}

	bool initScene();

	void debugMode();

	camera mCamera;
	cubemap mCubeMap;

	std::vector<meshNode> maconfig;
	
	//Primitives ou anciens systemes (à retirer à long terme)

	float						ambientLight;

	std::vector<sphere>			sphereContainer;
	std::vector<material>		materialContainer;
	std::vector<matRef>			materialRef;
	std::vector<light>			lightContainer;

	std::list<mesh>				meshes;

	nodeCuda * pRootBSP;

	triangle * pMemCuda;

    int sizex, sizey;
};



#endif // __SCENE_H
