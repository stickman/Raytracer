#include "rayTracer.h"

void
launch_cudaProcess(dim3 grid, dim3 block, float* g_odata, 
									int imgw, int imgh,
									point positionCamera,
									vecteur vXAxis,	vecteur vYAxis,vecteur vZAxis,
									std::vector<material> vector1, 
									std::vector<sphere> vector2 , 
									std::vector<light> vector4 ,
									cubemap cm,
									nodeCuda * root,triangle * memoryCuda)
{
	
	material * materials;
	cudaMalloc((void**) &materials, sizeof(material) * vector1.size());
	cutilCheckMsg("cudaMalloc failled\n.");

	for (int i = 0; i< vector1.size(); i++)
		cudaMemcpy((materials + i),&vector1[i],sizeof(material),cudaMemcpyHostToDevice);
		//*(materials + i) = vector1[i];

	cutilCheckMsg("cudaMemCpy failled\n.");

	sphere * spheres;
	cudaMalloc((void**) &spheres,sizeof(sphere) * vector2.size());
	cutilCheckMsg("cudaMalloc failled\n.");

	for (int i = 0; i< vector2.size(); i++)
		cudaMemcpy((spheres + i),&vector2[i],sizeof(sphere),cudaMemcpyHostToDevice);
		//*(spheres + i) = vector2[i];	

	cutilCheckMsg("cudaMemCpy failled\n.");

	
	light * lights;
	cudaMalloc((void**) &lights, sizeof(light) * vector4.size());
	cutilCheckMsg("cudaMalloc failled\n.");

	for (int i = 0; i< vector4.size(); i++)
		cudaMemcpy((lights + i),&vector4[i],sizeof(light),cudaMemcpyHostToDevice);
		//*(lights + i) = vector4[i];	
	cutilCheckMsg("cudaMemCpy failled\n.");

	//sbytes est la taille de la mémoire partagée
	int sbytes = 3 * sizeof(float) * 3;
    cudaProcess<<< grid, block, sbytes >>> (g_odata,	positionCamera,
														vXAxis,	vYAxis,vZAxis,
														materials, vector1.size(),
														spheres, vector2.size(),														
														lights, vector4.size(),	
														cm,
														root,memoryCuda);
														   
	cudaFree(	materials);
	cudaFree(	spheres);
	cudaFree(	lights);

}


nodeCuda* parcours (node * root, nodeCuda * memoryPere)
{
	nodeCuda * actuel;
	cudaMalloc((void**) &actuel,sizeof(nodeCuda));

	nodeCuda temp;

	temp.min = root->min;
	temp.max = root->max;
	temp.gauche = NULL;
	temp.droit = NULL;
	temp.parent = memoryPere;

	for (int i = 0; i < 16;i++)
	{
		temp.tabIdTriangle[i]=-1;
	}

	if (root->gauche)
	{
		nodeCuda * gauche = parcours(root->gauche,actuel);
		nodeCuda * droit = parcours(root->droit,actuel);

		temp.gauche = gauche;
		temp.droit = droit;
	}
	else
	{

		for (int i = 0; i < root->triangles->_usedSize;i++)
		{
			temp.tabIdTriangle[i]=root->triangles->getElement(i)->id;
		}
	}

	cudaMemcpy(actuel,&temp,sizeof(nodeCuda),cudaMemcpyHostToDevice);
	return actuel;

}

nodeCuda * build_BSP(std::list<mesh> meshes, triangle ** memoryCuda)
{
	int nbTriangles=0;

	std::list<mesh>::iterator it = meshes.begin();

	std::list<triangle>::iterator triangleIt;

	while (it!=meshes.end())
	{
		nbTriangles += (*it).triangles.size();
		it++;
	}

	triangle * trianglesCPU;

	cudaMalloc((void**)memoryCuda,sizeof(triangle) * nbTriangles);

	trianglesCPU = (triangle * ) malloc(sizeof(triangle) * nbTriangles);

	cutilCheckMsg("cudaMalloc failled\n.");
	int compteur = 0;

	for (it =  meshes.begin(); it!= meshes.end(); it++)
	{
		for ( triangleIt =  (*it).triangles.begin(); triangleIt != (*it).triangles.end(); triangleIt++)
		{
			triangleIt->id =compteur;
			cudaMemcpy((*memoryCuda + compteur),&(*triangleIt),sizeof(triangle),cudaMemcpyHostToDevice);
			memcpy((trianglesCPU+compteur),&(*triangleIt),sizeof(triangle));
			compteur++;
		}
	}
	
	node * rootCPU = buildBSP(trianglesCPU, nbTriangles);


	nodeCuda* retour = parcours(rootCPU,NULL);



	free(trianglesCPU);	

	cutilCheckMsg("Feel good.\n");


	return retour;

}

