#ifndef _BSP_TREE_H
#define _BSP_TREE_H

#include "Def.h"
#include "triangle.h"

struct side
{
	point	p;
	vecteur normal;
};

struct node
{
	node* parent;
	node* droit;
	node* gauche;
	listTriangle * triangles;
	point	min;
	point	max;

	node()
	{
		triangles = new listTriangle();
		triangles->init();

		parent = NULL;
		droit = NULL;
		gauche = NULL;
		min.x= 99999.0;		min.y=99999.0;		min.z=99999.0;
		max.x= -99999.0;	max.y=-99999.0;		max.z=-99999.0;
	}

};

struct nodeCuda
{
	nodeCuda* parent;
	nodeCuda* droit;
	nodeCuda* gauche;
	int		tabIdTriangle[16];
	point	min;
	point	max;

};


struct listPNode
{
	node ** pTeteTabNode;

	unsigned int _allocatedSize;
	unsigned int _usedSize;
	unsigned int _pointedNode;

	listPNode()
	{
		_allocatedSize = 1;
		_usedSize = 0;
		_pointedNode=0;

		pTeteTabNode = (node**) malloc (_allocatedSize * sizeof(node*) );		
	}

	void clear()
	{
		//if (pTeteTabNode) free (pTeteTabNode);
		_usedSize = 0;
		_pointedNode=0;

	}

	~listPNode()
	{
		if (pTeteTabNode) free (pTeteTabNode);
	}

	node * getNextElement ()
	{		
		node * retour = NULL;
		if (_pointedNode<_usedSize)
		{
			retour =  *(pTeteTabNode + _pointedNode);
			_pointedNode++;
		}
		return retour;
	}

	void  initIterator ()
	{		
		_pointedNode = 0;
	}


	void addNode(const node * n)
	{
		_usedSize++;
		if ( _usedSize > _allocatedSize)
		{
			_allocatedSize*=2;	
			pTeteTabNode = (node**) realloc (pTeteTabNode,_allocatedSize * sizeof(node*));
		}
		memcpy( pTeteTabNode + (_usedSize-1), &n ,sizeof(node*));

	}


	void removeNode(const node * n)
	{
		unsigned int rang =0;

		while ( ( *(pTeteTabNode + rang) != n) && rang <= _usedSize)
		{
			rang++;
		}

		if ((_usedSize == 0) || (rang > _usedSize)) return; 

		unsigned int nbElementSuivant = _usedSize - rang - 1;

		node ** temp = (node **) malloc ( sizeof(node*) * nbElementSuivant );

			memcpy(temp, (pTeteTabNode + rang +1 ) ,sizeof(node*) * nbElementSuivant );

			memcpy(pTeteTabNode + rang ,temp,sizeof(node*) * nbElementSuivant );

		free (temp);
		
		_usedSize--;

		if (_pointedNode!= 0) _pointedNode--;
		
	}

	bool empty()
	{
		return (_usedSize == 0);
	}

	void append(const listPNode & liste)
	{		
		for (unsigned int i =0 ; i < liste._usedSize ; i++)
		{
			addNode( *(liste.pTeteTabNode + i) );
		}
	}


};





node* buildBSP(triangle * listT, unsigned int NbrT);

#endif