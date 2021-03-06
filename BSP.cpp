#include "BSP.h"

#include "Triangle_AABB.h"

void PROCESSBSPNODES(listPNode & activelist, listPNode & nextlist)
{
	activelist.initIterator();
	node * i = activelist.getNextElement();

	while (i!=NULL)
	{	
		if (i->min.x == 99999.0) //initialisation rootnode
		{
			i->triangles->initIterator();
			triangle * t = i->triangles->getNextElement();
			while(t != NULL)
			{

				if (i->min.x > t->min.x)
					i->min.x = t->min.x;

				if (i->min.y > t->min.y)
					i->min.y = t->min.y;

				if (i->min.z > t->min.z)
					i->min.z = t->min.z;

				if (i->max.x < t->max.x)
					i->max.x = t->max.x;

				if (i->max.y < t->max.y)
					i->max.y = t->max.y;

				if (i->max.z < t->max.z)
					i->max.z = t->max.z;

				t = i->triangles->getNextElement();
			}
		}
		i = activelist.getNextElement();

	}//fin while

	activelist.initIterator();
	i = activelist.getNextElement();
	while (i!=NULL)
	{
		/*side cotesBox[6];
		cotesBox[0].p = i->min;		cotesBox[1].p = i->max;
		cotesBox[0].normal.x = -1;		cotesBox[1].normal.x = 1;
		cotesBox[0].normal.y = 0;		cotesBox[1].normal.y = 0;
		cotesBox[0].normal.z = 0;		cotesBox[1].normal.z = 0;
		cotesBox[2].p = i->min;		cotesBox[3].p = i->max;
		cotesBox[2].normal.x = 0;		cotesBox[3].normal.x = 0;
		cotesBox[2].normal.y = -1;		cotesBox[3].normal.y = 1;
		cotesBox[2].normal.z = 0;		cotesBox[3].normal.z = 0;
		cotesBox[4].p = i->min;		cotesBox[5].p = i->max;
		cotesBox[4].normal.x = 0;		cotesBox[5].normal.x = 0;
		cotesBox[4].normal.y = 0;		cotesBox[5].normal.y = 0;
		cotesBox[4].normal.z = -1;		cotesBox[5].normal.z = 1;

		for (int iBox = 0; iBox < 6 ; iBox++)
		{
			//REDUCTION DE BOX ICI
		}
		*/
		//SPLIT

		vecteur v = i->max - i->min ;

		node *filsG = new node();
		node *filsD = new node();

		//filsG->init();
		//filsD->init();

		i->gauche = filsG;
		i->droit  = filsD;

		i->gauche->parent =i;
		i->droit->parent = i;

		i->gauche->min = i->min;
		i->gauche->max = i->max;

		i->droit->min = i->min;
		i->droit->max = i->max;

		if ( (v.x > v.y) && (v.x >v.z))
		{
			//on coupe selon YZ
			i->gauche->max.x = (i->min.x + i->max.x) /2.0;
			i->droit->min.x = (i->min.x + i->max.x) /2.0;

		}
		else
		{
			if (v.y >v.z)
			{
				//on coupe selon XZ
				i->gauche->max.y = (i->min.y + i->max.y) /2.0;
				i->droit->min.y = (i->min.y + i->max.y) /2.0;

			}
			else
			{
				//on coupe selon XY

				i->gauche->max.z = (i->min.z + i->max.z) /2.0;
				i->droit->min.z = (i->min.z + i->max.z) /2.0;

			}
		}

		nextlist.addNode(i->gauche);
		nextlist.addNode(i->droit);
		
		i->triangles->initIterator();

		triangle * pTriangleTemp = i->triangles->getNextElement();

		while (pTriangleTemp!=NULL)// (unsigned int iTriangle =0 ; iTriangle < (maChunkList.list + iChunk)->nbTriangles ; iTriangle++)
		{				

			float tab[3][3];
			tab[0][0] = pTriangleTemp->A.x;tab[0][1] = pTriangleTemp->A.y;tab[0][2] = pTriangleTemp->A.z;
			tab[1][0] = pTriangleTemp->B.x;tab[1][1] = pTriangleTemp->B.y;tab[1][2] = pTriangleTemp->B.z;
			tab[2][0] = pTriangleTemp->C.x;tab[2][1] = pTriangleTemp->C.y;tab[2][2] = pTriangleTemp->C.z;

			float centerG[3];float halfsizeG[3];
			centerG[0] = (i->gauche->max.x + i->gauche->min.x)*0.5 ; halfsizeG[0] =( i->gauche->max.x - i->gauche->min.x )*0.5 ;
			centerG[1] = (i->gauche->max.y + i->gauche->min.y)*0.5 ; halfsizeG[1] =( i->gauche->max.y - i->gauche->min.y )*0.5 ;
			centerG[2] = (i->gauche->max.z + i->gauche->min.z)*0.5 ; halfsizeG[2] =( i->gauche->max.z - i->gauche->min.z )*0.5 ;

			float centerD[3];float halfsizeD[3];
			centerD[0] = (i->droit->max.x + i->droit->min.x)*0.5 ; halfsizeD[0] =( i->droit->max.x - i->droit->min.x )*0.5;
			centerD[1] = (i->droit->max.y + i->droit->min.y)*0.5 ; halfsizeD[1] =( i->droit->max.y - i->droit->min.y )*0.5 ;
			centerD[2] = (i->droit->max.z + i->droit->min.z)*0.5 ; halfsizeD[2] =( i->droit->max.z - i->droit->min.z )*0.5 ;

			if (triBoxOverlap(centerG,halfsizeG,tab))
			{
				if (triBoxOverlap(centerD,halfsizeD,tab))
				{
					i->gauche->triangles->addTriangle(pTriangleTemp);
					i->droit->triangles->addTriangle(pTriangleTemp);
				}
				else
				{
					i->gauche->triangles->addTriangle(pTriangleTemp);
				}
			}
			else
			{					
				i->droit->triangles->addTriangle(pTriangleTemp);	
			}

			pTriangleTemp = i->triangles->getNextElement();			
		}
		
		i = activelist.getNextElement();
	}

	nextlist.initIterator();
	node* ch = nextlist.getNextElement();
	while (ch!=NULL)
	{
		if ( ch->triangles->_usedSize <= 16)
		{
			nextlist.removeNode(ch);
		}
		ch=nextlist.getNextElement();
	}

}


node* buildBSP(triangle * listT, unsigned int NbrT)
{

	for (unsigned int i = 0; i< NbrT; i++)
	{
		(listT+i)->SetMinMaxBound();
	}


	listPNode * nextList = new listPNode();//temp

	listPNode* activeList = new listPNode();//temp


	node * rootNode = new node();

	rootNode->triangles->append(listT,NbrT);

	activeList->addNode(rootNode);

	while (!activeList->empty())
	{		
		nextList->clear();

		PROCESSBSPNODES(*activeList, *nextList);

		listPNode * ptrList = activeList;
		activeList = nextList;
		nextList = ptrList;

	}




	//ICI construction KDTree sur 
	delete (nextList);
	delete (activeList);
	return rootNode;

}