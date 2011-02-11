#include "Matrix.h"

matrice33 operator  * (const float& f,const matrice33 &m)
{
	matrice33 ret;
	for (int i = 0; i<9;i++)
	{
		ret.tab[i] = m.tab[i] * f;
	}
	return ret;
}

matrice33 operator  + (const matrice33 &m1,const matrice33 &m2)
{	   
	matrice33 ret;

	for (int i = 0; i<9;i++)
	{
		ret.tab[i] = m1.tab[i] + m2.tab[i] ;
	}

	return ret;
}

matrice33 operator  * (const matrice33 &m1,const matrice33 &m2)
{	   
	matrice33 ret;

	for (int i = 0; i<3;i++)
	{		
		for (int j = 0; j<3;j++)
		{
			ret.tab[i + j*3] = m1.tab[j*3 ] + m2.tab[i] + m1.tab[1 + j*3] + m2.tab[ 3 + i] +m1.tab[2 + j*3 ] + m2.tab[6 + i];
		}
	}

	return ret;
}






vecteur operator  * (const matrice33 &m,const vecteur &v)
{	   
	vecteur ret;

	ret.x = v.x * m.tab[0] + v.y * m.tab[1] + v.z * m.tab[2];
	ret.y = v.x * m.tab[3] + v.y * m.tab[4] + v.z * m.tab[5];
	ret.z = v.x * m.tab[6] + v.y * m.tab[7] + v.z * m.tab[8];

	return ret;
}




