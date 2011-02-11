#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "def.h"

struct triangle
{
	point A;
	point B;
	point C;

	point min;
	point max;

	char * materialName;

	int id;

	vecteur n;//on stock les normales, on gagne beaucoup en perf!

	int materialId;

	void calcNormal ()
	{		
		/*
		LES TRIANGLES DOIVENT ETRE ENTRES EN SENS TRIGO
			  A					  A	
			/ + \  on voit		/ . \  on voit pas!
		   B ----C			   C ----B 
		*/	

		vecteur AB = this->B - this->A;
		vecteur AC = this->C - this->A;
		this->n = AB^AC;
		this->n.normalise();
    }

	void SetMinMaxBound()
	{
		min.x = A.x;
		min.y = A.y;
		min.z = A.z;

		max.x = A.x;
		max.y = A.y;
		max.z = A.z;

		if (B.x < min.x) min.x = B.x;

		if (C.x < min.x) min.x = C.x;

		if (B.y < min.y) min.y = B.y;

		if (C.y < min.y) min.y = C.y;

		if (B.z < min.z) min.z = B.z;

		if (C.z < min.z) min.z = C.z;



		if (B.x > max.x) max.x = B.x;

		if (C.x > max.x) max.x = C.x;

		if (B.y > max.y) max.y = B.y;

		if (C.y > max.y) max.y = C.y;

		if (B.z > max.z) max.z = B.z;

		if (C.z > max.z) max.z = C.z;

	}

	bool appartient(point pMin, point pMax)
	{
		return false;

	}

};



struct listTriangle
{

	triangle* pTriangleTete;

	unsigned int _allocatedSize;
	unsigned int _usedSize;

	unsigned int _pointedTriangle;

	void init()
	{
		_allocatedSize = 1;
		_usedSize = 0;
		_pointedTriangle=0;

		pTriangleTete = (triangle*) malloc (sizeof(triangle) *_allocatedSize );
	}
	
	triangle * getNextElement ()
	{		
		triangle * retour = NULL;
		if (_pointedTriangle<_usedSize)
		{
			retour =  pTriangleTete + _pointedTriangle;
			_pointedTriangle++;
		}
		return retour;
	}


	triangle * getElement (unsigned int nb)
	{		
		if (_usedSize<nb)
		{
			return NULL;
		}
		else
		{
			return pTriangleTete + nb;
		}
	}


	void initIterator ()
	{		
		_pointedTriangle = 0;
	}


	void addTriangle(triangle * t)
	{
		_usedSize++;
		if ( _usedSize > _allocatedSize)
		{
			_allocatedSize*=2;
			pTriangleTete = (triangle*) realloc(pTriangleTete,_allocatedSize * sizeof(triangle));
		}
		memcpy( pTriangleTete + (_usedSize-1), t ,sizeof(triangle));


	}

	bool empty()
	{
		return (_usedSize == 0);
	}

	void append(const listTriangle& liste)
	{		
		for (unsigned int i =0 ; i < liste._usedSize ; i++)
		{
			addTriangle( liste.pTriangleTete + i);
		}
	}

	void append(triangle * pTriangles, unsigned int nb)
	{		
		_usedSize += nb;
		bool bRealoc = _usedSize > _allocatedSize;
		while (_usedSize > _allocatedSize )
		{
			_allocatedSize*=2;			
		}
		if (bRealoc)
		{
			pTriangleTete = (triangle*) realloc(pTriangleTete,_allocatedSize * sizeof(triangle));
		}
		
		memcpy( pTriangleTete + (_usedSize-nb), pTriangles ,nb * sizeof(triangle));
		
	}

	void clear()
	{
		pTriangleTete = NULL;
		init();
	}


};


long t_c_intersection(const triangle & t,const point & min, const point & max);

#endif