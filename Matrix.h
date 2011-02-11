#ifndef __MATRIX__
#define __MATRIX__

#include "Def.h"

struct matrice33
{	
	float tab[9];
	// @ du tableau :
	/*
	0	1	2
	3	4	5
	6	7	8

	*/
	matrice33& operator += (const matrice33 &m)
	{
		for (int i = 0; i<9;i++)
		{
			tab[i] += m.tab[i] ;
		}
	    return *this;
    }


	matrice33 operator  * (const float& f)
	{
	for (int i = 0; i<9;i++)
	{
		tab[i] *= f;
	}

	return *this;
	}

};

matrice33 operator  * (const float& f,const matrice33 &m);
matrice33 operator  + (const matrice33 &m1,const matrice33 &m2);
matrice33 operator  * (const matrice33 &m1,const matrice33 &m2);
vecteur operator  * (const matrice33 &m,const vecteur &v);

static float matrice33_0[9]= {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}; 
static float matrice33_Id[9] = {1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};



struct matrice44
{	
	float tab[16];
	// @ du tableau :
	/*
	0	1	2	3
	4	5	6	7
	8	9	8	11
	12	13	14	15
	*/
	matrice44& operator += (const matrice44 &m)
	{
		for (int i = 0; i<16;i++)
		{
			tab[i] += m.tab[i] ;
		}
	    return *this;
    }


	matrice44 operator  * (const float& f)
	{
		for (int i = 0; i<16;i++)
		{
			tab[i] *= f;
		}

		return *this;
	}

};




static float matrice44_0[16]= {0.0,0.0,0.0,0.0,
								0.0,0.0,0.0,0.0,
								0.0,0.0,0.0,0.0,
								0.0,0.0,0.0,0.0};

static float matrice44_Id[16] = {1.0,0.0,0.0,0.0,
								0.0,1.0,0.0,0.0,
								0.0,0.0,1.0,0.0,
								0.0,0.0,0.0,1.0}; 



#endif
