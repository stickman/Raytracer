#include "ColladaLoader.h"

using namespace std;

void parse_index(string & int_array,vector<int> & tokens)
{
	printf ("\t\t\t debug_info : Parse index\n");

	string accumulator;

	while (int_array.size()!=0)
	{
		switch (int_array[0])
		{ 
			//the first letter of equation
			case ' ':
			{
				int_array.erase(0, 1);       //remove the ' '
				size_t siz;

				for (siz = 0; siz != int_array.size(); ++siz)
				{
					if (int_array[siz] == ' ') break;
				}

				tokens.push_back(atoi(accumulator.c_str()));
				accumulator.clear();

				string temp = int_array.substr(0, siz);
				
				break;
			}

			default:
			{
				accumulator += int_array.substr(0, 1);
				int_array.erase(0, 1);
				break;
			}
		}
	}
	if (accumulator.length()>0)
		tokens.push_back(atoi(accumulator.c_str()));

	printf ("\t\t\t debug_info : index Parsed\n");

}




void parse_float_array(string & float_array,vector<float> & tokens)
{
	printf ("\t\t\t debug_info : Parse array\n");

	string accumulator;

	int debugStep=0;

	while (float_array.size()!=0)
	{

		switch (float_array[0])
		{ 
			//the first letter of equation
			case ' ':
			{				
				//remove the ' '
				//printf ("On va enlever le ' '\n");
				float_array.erase(0, 1);
				//printf ("On a enlevé le ' '\n");
				if (accumulator.length()>0)
				{
					//printf ("On va calculer la valeur\n");
					float temp = atof(accumulator.c_str());
					//printf ("On va ajouter la valeur a la liste\n");
					tokens.push_back(temp);
					//printf ("On a ajouté la valeur a la liste, on va nettoyer l'accumulateur\n");
					accumulator.clear();
					//printf ("On a nettoyé l'accumulateur\n");
				}
								
				break;
			}

			default:
			{
				//printf ("On va ajouter un truc dans l'accumulateur\n");
				accumulator += float_array.substr(0, 1);
				//printf ("On a ajouté un truc dans l'accumulateur, on l'enleve de la string d'origine\n");
				float_array.erase(0, 1);
				//printf ("On a enlevé le truc de la string d'origine\n");
				break;
			}
		}

	}

	if (accumulator.length()>0)
		tokens.push_back(atof(accumulator.c_str()));

	printf ("\t\t\t debug_info : array Parsed\n");
}



CColladaLoader CColladaLoader::m_instance;


bool CColladaLoader::loadCollada(const char * fileName, CScene &scene)
{	
	printf("\nLoading file : %s\n", fileName);

	TiXmlDocument doc;// = new TiXmlDocument();

	scene.sizex = 640;
	scene.sizey = 480;

	bool retour = doc.LoadFile(fileName);

	if (retour)
	{

		TiXmlElement * rootElement = doc.RootElement();

		retour = retour && (rootElement!=NULL);

		TiXmlElement *pElem = rootElement->FirstChildElement("library_visual_scenes");

		if (pElem!=NULL) 
		{	
			retour && _loadNodes(pElem,scene.maconfig);
		}
		else
		{
			printf("\tFile %s has no valid visual scene library.\n" , fileName);
		}

		loadMaterials(rootElement,scene); 

		TiXmlElement *pElemGeo = rootElement->FirstChildElement("library_geometries");

		if (pElemGeo) 
		{			
			retour = retour && (pElemGeo!=NULL);

			retour = retour && loadGeometryScene(pElemGeo,scene);
		}
		else
		{
			printf("\tFile %s has no valid library geometry.\n" , fileName);
		}


	}
	else
	{
		printf("\tFile %s not found or not readable.\n" , fileName);
	}

	printf(retour ? "File %s loaded.\n":"File %s not loaded\n" , fileName);

	return retour;

}


bool CColladaLoader::loadGeometryScene(TiXmlElement * rootLibGeometry, CScene &scene)
{
	printf("\tScene Geometry loading\n");

	bool retour = true;

	TiXmlElement * pGeometry = rootLibGeometry->FirstChildElement("geometry");

	while (pGeometry && retour)
	{
		mesh newMesh;
	
		newMesh.name = (char * ) malloc (sizeof(char) * strlen(pGeometry->Attribute("name")));

		strcpy(newMesh.name,pGeometry->Attribute("name"));

		printf("\t mesh %s in loading\n", newMesh.name);

		vector<meshNode>::iterator it = scene.maconfig.begin();

		while (it!=scene.maconfig.end())
		{
			if (strcmp(it->name.c_str(),newMesh.name)==0)
			{
				retour = retour && loadMesh(pGeometry,newMesh,it->Quaternion,scene);
				break;
			}
			it++;
		}

		if (it==scene.maconfig.end())
		{
			matrice44 mID;
			memcpy(mID.tab,matrice44_Id,sizeof(float) *16);
			retour = retour && loadMesh(pGeometry,newMesh,mID,scene);		
		}

		printf("\t mesh %s loaded\n", newMesh.name);

		scene.meshes.push_back(newMesh);

		pGeometry= pGeometry->NextSiblingElement("geometry");

	}

	printf(retour ?"\tScene Geometry loading\n" :"\tScene Geometry loading FAILED\n" );

	return retour;

}



bool CColladaLoader::_loadNodes(TiXmlElement * matrixRoot, std::vector<meshNode> & meshNodes)
{
	
	printf("\nVisual scene loading\n");

	bool retour = true;

	TiXmlElement * pVisual_scene = matrixRoot->FirstChildElement("visual_scene");

	while (pVisual_scene && retour)
	{
		printf("\t node %s in loading\n", pVisual_scene->Attribute("name"));

		TiXmlElement * pNode = pVisual_scene->FirstChildElement("node");

		while (pNode && retour)
		{
			meshNode newMesh;	

			TiXmlElement * pMatrix = pNode->FirstChildElement("matrix");

			if (pMatrix)
			{
				vector<float> values;
				
				const char * value= pMatrix->GetText();
		
				std::string text(value);
				
				parse_float_array(text,values);
				for (int i = 0 ; i<16;i++)
				{
					newMesh.Quaternion.tab[i]= values[i];
				}
			}

			TiXmlElement * pInstance = pNode->FirstChildElement("instance_geometry");

			if(pInstance)
			{				
				//printf("\t node %s loaded\n", pInstance->Attribute("url"));

				newMesh.name = std::string(pInstance->Attribute("url"));
				newMesh.name.erase(0,1);//"#"

				TiXmlElement * pMaterial = pInstance->FirstChildElement("bind_material");
				if(pMaterial)
				{
					TiXmlElement * pTempLvl1 = pMaterial->FirstChildElement("technique_common");
					if(pTempLvl1)
					{
						TiXmlElement * pTempLvl2 = pTempLvl1->FirstChildElement("instance_material");
						if(pTempLvl2)
						{
							newMesh.material = std::string (pTempLvl2->Attribute("target"));

							TiXmlElement * pTempLvl3 = pTempLvl2->FirstChildElement("bind_vertex_input");
							if(pTempLvl3)
							{

							}
						}
					}
				}

			}

			meshNodes.push_back(newMesh);

			pNode= pNode->NextSiblingElement("node");
		}

		pVisual_scene= pVisual_scene->NextSiblingElement("visual_scene");

	}

	printf(retour ?"\tVisual scene loading\n" :"\tVisual scene loading FAILED\n" );

	return retour;


}




bool CColladaLoader::_loadVertices(TiXmlElement * verticesRoot, vector<point> & Vertices,matrice44 & mat)
{
	printf("\t\t vertices in loading\n");

	bool bRetour = true;
	
	TiXmlElement * pArray =  verticesRoot->FirstChildElement("float_array");
	
	bRetour = bRetour && (pArray!=NULL);

	int nbVertices = 0;

	if (bRetour) pArray->Attribute("count",&nbVertices);

	bRetour = bRetour && (nbVertices>0);

	if (bRetour)
	{
		Vertices.resize(nbVertices / 3);// = (point *) malloc (sizeof(point) * nbVertices);
		
		int iVertice = 0;

		const char * value= pArray->GetText();
		
		string text(value);

		vector<float> verticesValues;

		verticesValues.reserve(nbVertices);

		parse_float_array(text,verticesValues);

		for (unsigned int iValue = 0; iValue< verticesValues.size();iValue++)
		{
			switch (iValue%3)
			{
			case (0):
				{
					Vertices[iVertice].x=  verticesValues[iValue] + mat.tab[3]; 
					break;
				}
			case (1):
				{
					Vertices[iVertice].y=  verticesValues[iValue] + mat.tab[11] ; 
					break;
				}
			case (2):
				{
					Vertices[iVertice].z=  verticesValues[iValue] + mat.tab[7] ; 

					/*
					ICI multiplication par quaternion pour scale et rotation!!!*/

					matrice33 MScaleRot;
					/*for (int i = 0; i<3;i++)
					{
						for (int j = 0; j<3;j++)
						{
							MScaleRot.tab[i+j*3]=mat.tab[i+j*4];
						}
						MScaleRot.tab[i+i*3]*=100.0;//blender 0.01->1
					}*/

					MScaleRot.tab[0]=-mat.tab[0];
					MScaleRot.tab[1]=-mat.tab[2];
					MScaleRot.tab[2]=-mat.tab[1];

					MScaleRot.tab[3]=-mat.tab[4];
					MScaleRot.tab[4]=-mat.tab[6];
					MScaleRot.tab[5]=-mat.tab[5];

					MScaleRot.tab[6]=-mat.tab[8];
					MScaleRot.tab[7]=-mat.tab[10];
					MScaleRot.tab[8]=-mat.tab[9];


					vecteur v;
					v.x= Vertices[iVertice].x;
					v.y= Vertices[iVertice].y;
					v.z= Vertices[iVertice].z;

					v = MScaleRot * v;

					Vertices[iVertice].x = v.x *100.0;
					Vertices[iVertice].y = v.y *100.0;
					Vertices[iVertice].z = v.z *100.0;				


					iVertice++;
					break;
				}
			default:
				{
					printf("\t\t ERROR\n");
					break;
				}
			}	
			
		}
		printf("\t\t vertices loaded\n");
	}
	return bRetour;
}

bool CColladaLoader::_loadNormals(TiXmlElement * normalsRoot, vector<vecteur> & Normals)
{
	printf("\t\t normals in loading\n");

	bool bRetour = true;
	
	TiXmlElement * pArray =  normalsRoot->FirstChildElement("float_array");
	
	bRetour = bRetour && (pArray!=NULL);

	int nbNormals = 0;

	if (bRetour) pArray->Attribute("count",&nbNormals);	
	
	bRetour = bRetour && (nbNormals>0);

	if (bRetour)
	{
		Normals.resize(nbNormals/3);

		int iNormal = 0;

		const char * value= pArray->GetText();

		std::string text(value);

		vector<float> normalsValues;

		normalsValues.reserve(nbNormals);

		parse_float_array(text,normalsValues);

		for (unsigned int iValue = 0; iValue< normalsValues.size();iValue++)
		{

			switch (iValue%3)
			{
			case (0):
				{
					Normals[iNormal].x=  normalsValues[iValue];
					break;
				}
			case (1):
				{
					Normals[iNormal].y=  normalsValues[iValue];
					break;
				}
			default :
				{
					Normals[iNormal].z=  normalsValues[iValue];
					iNormal++;
					
					break;
				}
			}
			
		}
		printf("\t\t normals loaded\n");
	}

	return bRetour;

}




bool CColladaLoader::_loadTriangles(TiXmlElement * pMesh,mesh & meshOut , vector<point> & Vertices,vector<vecteur> & Normals,CScene & scene)
{
	bool bRetour = true;

	printf("\t\t triangles of %s in loading\n", meshOut.name);

	TiXmlElement * pTriangles = pMesh->FirstChildElement("triangles");

	int nbTriangles = 0;	

	int offsetVertex = -1;
	int offsetNormal = -1;
	
	bRetour = bRetour && (pTriangles!=NULL);



	while (bRetour && (pTriangles!=NULL))
	{	
		if (bRetour) 	pTriangles->Attribute("count",&nbTriangles);

		TiXmlElement * pInput = pTriangles->FirstChildElement("input");
		int nbInput = 0;

		while (pInput)
		{
			nbInput++;
			if (strcmp(pInput->Attribute("semantic"),"VERTEX")==0)
			{
				offsetVertex = atoi(pInput->Attribute("offset"));
			}
			if (strcmp(pInput->Attribute("semantic"),"NORMAL")==0)
			{
				offsetNormal = atoi(pInput->Attribute("offset"));
			}
			pInput= pInput->NextSiblingElement("input");
		}

		

		bRetour = bRetour && (nbTriangles>0);

		TiXmlElement * pArray =  pTriangles->FirstChildElement("p");

		unsigned int materialID = 0;

		std::vector<matRef>::iterator itMat = scene.materialRef.begin();

		while(itMat != scene.materialRef.end())
		{
			if (pTriangles->Attribute("material") && strcmp(itMat->material.c_str(),pTriangles->Attribute("material"))==0)
				materialID= itMat->id;
			itMat++;
		}

		meshOut.triangles.resize(nbTriangles);

		vecteur NormaleTemp;
	
		int iValue = 0;

		const char * value= pArray->GetText();

		string Text(value);

		vector<int> trianglesIndex;

		trianglesIndex.reserve(nbTriangles);

		parse_index(Text,trianglesIndex);				

		std::list<triangle>::iterator it = meshOut.triangles.begin();

		while (!trianglesIndex.empty())
		{
			if ( (iValue%(3*nbInput)) == (0 + 0 * nbInput))
			{
				(*it).A = Vertices[ trianglesIndex.front()];			
			}
			if ( (iValue%(3*nbInput)) == (1 + 0 * nbInput))
			{
				NormaleTemp = Normals[trianglesIndex.front()];
			}
			if ( (iValue%(3*nbInput)) == (0 + 1 * nbInput))
			{
				(*it).B = Vertices[ trianglesIndex.front()];
			}
			if ( (iValue%(3*nbInput)) == (1 + 1 * nbInput))
			{
				NormaleTemp = Normals[trianglesIndex.front()];
			}
			if ( (iValue%(3*nbInput)) == (0 + 2 * nbInput))
			{
				(*it).C = Vertices[ trianglesIndex.front()];
			}
			if ( (iValue%(3*nbInput)) == (1 + 2 * nbInput))
			{
				NormaleTemp = Normals[trianglesIndex.front()];

				/*(*it).n.x = NormaleTemp.x;
				(*it).n.y = NormaleTemp.y;
				(*it).n.z = NormaleTemp.z;*/

				it->calcNormal();

				it->materialId = materialID;

				it++;						
			}
			trianglesIndex.erase(trianglesIndex.begin());
			iValue++;
		}//FIN FOR

	pTriangles = pTriangles->NextSiblingElement("triangles");	
		
	}
		
	printf("\t\t triangles of %s loaded\n", meshOut.name);

	return bRetour;

}




bool CColladaLoader::loadMesh(TiXmlElement * rootGeometry, mesh &meshOut,matrice44 & mat,CScene & scene)
{
	bool bRetour = true;

	TiXmlElement * pMesh;

	pMesh = rootGeometry->FirstChildElement("mesh");
	
	bRetour = bRetour && (pMesh!=NULL);

	if (bRetour)
	{
		vector<point>		Vertices;
		vector<vecteur>		Normals;

		char * nameVertices = (char *) calloc ((strlen (meshOut.name) + strlen("-Position")),sizeof(char));
		char * nameNormals  = (char *) calloc ((strlen (meshOut.name) + strlen("-Normals")),sizeof(char));

		strcat(nameVertices,meshOut.name);
		strcat(nameNormals,meshOut.name);
		strcat(nameVertices,"-Position");
		strcat(nameNormals,"-Normals");

		TiXmlElement * pTemp = pMesh->FirstChildElement("source");

		while (pTemp && bRetour)
		{
			if (strcmp(pTemp->Attribute("id"),nameVertices)==0)
			{
				bRetour = bRetour && _loadVertices(pTemp,Vertices,mat);				
			}

			if (strcmp(pTemp->Attribute("id"),nameNormals)==0)
			{
				bRetour = bRetour && _loadNormals(pTemp,Normals);
			}
			pTemp= pTemp->NextSiblingElement("source");
		}

		//delete (nameVertices);
		//delete (nameNormals);

		if (bRetour)
		{			
			bRetour = bRetour && _loadTriangles(pMesh,meshOut,Vertices,Normals,scene);
		}
	}

	return bRetour;

}


bool CColladaLoader::loadLights(TiXmlElement * rootGeometry, CScene &scene)
{

	return true;

}

bool CColladaLoader::loadMaterials(TiXmlElement * rootGeometry,  CScene &scene)
{
	//On recupere les id mat et leur FX liés
	TiXmlElement * pTemp = rootGeometry->FirstChildElement("library_materials");

	if (pTemp!=NULL)
	{
		TiXmlElement * pMaterial = pTemp->FirstChildElement("material");		
		while(pMaterial!=NULL)
		{
			matRef mRef;
			material m;

			mRef.material.clear();
			mRef.material.append(std::string(pMaterial->Attribute("id")));

			scene.materialContainer.push_back(m);
			mRef.id = scene.materialContainer.size();



			TiXmlElement * pEffect= pMaterial->FirstChildElement("instance_effect");	
			if(pEffect!=NULL)
			{
				mRef.materialFX.clear();
				mRef.materialFX.append(std::string(pEffect->Attribute("url")));
				mRef.materialFX.erase(0,1);
			}

			scene.materialRef.push_back(mRef);
			pMaterial = pMaterial->NextSiblingElement("material");
		}
	}

	pTemp = rootGeometry->FirstChildElement("library_effects");

	if (pTemp!=NULL)
	{
		TiXmlElement * pEffect = pTemp->FirstChildElement("effect");
		while (pEffect!=NULL)
		{
			std::vector<matRef>::iterator it = scene.materialRef.begin();
			std::vector<material>::iterator itMat = scene.materialContainer.begin();
			itMat++;//DEFAUT material
			while(it!=scene.materialRef.end())
			{
				if(strcmp(it->materialFX.c_str(),pEffect->Attribute("id"))==0)
				{

					break;
				}				
				it++;
				itMat++;
			}

			std::vector<float> temp;

			TiXmlElement *pPhong = pEffect->FirstChildElement("profile_COMMON")->FirstChildElement("technique")->FirstChildElement("phong");
			if (pPhong!=NULL)
			{
				std::string valueAmbient(pPhong->FirstChildElement("ambient")->FirstChildElement("color")->GetText());
				parse_float_array(valueAmbient,temp);
				itMat->ambient.red = temp[0];itMat->ambient.green = temp[1];itMat->ambient.blue = temp[2];
				temp.clear();

				std::string valueDiffuse(pPhong->FirstChildElement("diffuse")->FirstChildElement("color")->GetText());
				parse_float_array(valueDiffuse,temp);				
				itMat->diffuse.red = temp[0];itMat->diffuse.green = temp[1];itMat->diffuse.blue = temp[2];
				temp.clear();

				std::string valueSpecular(pPhong->FirstChildElement("specular")->FirstChildElement("color")->GetText());
				parse_float_array(valueSpecular,temp);				
				itMat->specular.red = temp[0];itMat->specular.green = temp[1];itMat->specular.blue = temp[2];
				temp.clear();

				itMat->shininess = atof(pPhong->FirstChildElement("shininess")->FirstChildElement("float")->GetText());
				
				itMat->reflectivity = atof(pPhong->FirstChildElement("reflectivity")->FirstChildElement("float")->GetText());

				itMat->transparency = atof(pPhong->FirstChildElement("transparency")->FirstChildElement("float")->GetText());

			}
			else
			{
				printf("Erreur dans le chargement de materiaux\n");
				return false;
			}

			pEffect= pEffect->NextSiblingElement("effect");
		}

	}
	return true;

}

bool CColladaLoader::loadCamera(TiXmlElement * rootCamera,  CScene &scene)
{

	TiXmlElement * pTemp = rootCamera->FirstChildElement("yfov");

	bool bRetour = (pTemp!=NULL);
	if(bRetour)
	{		
	}

	return true;

}