#ifndef __CColladaLoader_H__
#define __CColladaLoader_H__

#include "TinyXML/tinyxml.h"

#include "Scene.h"
#include "Matrix.h"

class CColladaLoader
{

	bool loadGeometryScene(TiXmlElement * rootMesh, CScene &scene);

	bool loadMesh(TiXmlElement * rootMesh, mesh &meshOut,matrice44 & mat,CScene &);

	bool CColladaLoader::loadCamera(TiXmlElement * rootGeometry, CScene &scene);

	bool CColladaLoader::loadLights(TiXmlElement * rootGeometry, CScene &scene);

	bool CColladaLoader::loadMaterials(TiXmlElement * rootGeometry, CScene &scene);
	
	bool CColladaLoader::_loadNodes(TiXmlElement * matrixRoot, std::vector<meshNode> & meshNodes);

	static CColladaLoader m_instance;

	bool _loadVertices(TiXmlElement * verticesRoot, std::vector<point> & vertices,matrice44 & mat);

	bool _loadNormals(TiXmlElement * normalsRoot, std::vector<vecteur> & normals);

	bool _loadTriangles(TiXmlElement * normalsRoot,mesh & meshOut , std::vector<point> & Vertices, std::vector<vecteur> & Normals,CScene & scene);

public:

	CColladaLoader(void){}

	~CColladaLoader(void){}

	static CColladaLoader * getSingletonPtr(){return &m_instance;}

	bool loadCollada(const char * fileName,CScene &scene);


};

#endif
