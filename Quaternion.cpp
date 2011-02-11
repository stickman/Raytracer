#include "Quaternion.h"


void YAW(vecteur & vX,vecteur & vY, vecteur & vZ, float angle)
{
	matrice33 M,I,M1,M2;

	memcpy(M.tab,matrice33_0,9 * sizeof(float));

	memcpy(I.tab,matrice33_Id,9 * sizeof(float));

	M1.tab[0] = vY.x * vY.x;	M1.tab[1] = vY.y * vY.x;	M1.tab[2] = vY.z * vY.x;
	M1.tab[3] = vY.x * vY.y;	M1.tab[4] = vY.y * vY.y;	M1.tab[5] = vY.z * vY.y;
	M1.tab[6] = vY.x * vY.z;	M1.tab[7] = vY.y * vY.z;	M1.tab[8] = vY.z * vY.z;

	M2.tab[0] =  0;		M2.tab[1] = -vY.z;	M2.tab[2] =  vY.y;
	M2.tab[3] =  vY.z;	M2.tab[4] =  0;		M2.tab[5] = -vY.x ;
	M2.tab[6] = -vY.y;	M2.tab[7] =  vY.x;	M2.tab[8] =  0;


	M = cos (angle) * I + (1- cos (angle)) * M1 + sin (angle) * M2;

	vecteur temp = vX;

	vX = M *  temp;
	//vX.normalise();

	vZ = vX^vY;
	//vZ.normalise();
}

void roll(vecteur & vX,vecteur & vY, vecteur & vZ, float angle)
{
	matrice33 M,I,M1,M2;

	memcpy(M.tab,matrice33_0,9 * sizeof(float));
	memcpy(I.tab,matrice33_Id,9 * sizeof(float));

	M1.tab[0] = vX.x * vX.x;	M1.tab[1] = vX.y * vX.x;	M1.tab[2] = vX.z * vX.x;
	M1.tab[3] = vX.x * vX.y;	M1.tab[4] = vX.y * vX.y;	M1.tab[5] = vX.z * vX.y;
	M1.tab[6] = vX.x * vX.z;	M1.tab[7] = vX.y * vX.z;	M1.tab[8] = vX.z * vX.z;

	M2.tab[0] =  0;		M2.tab[1] = -vX.z;	M2.tab[2] =  vX.y;
	M2.tab[3] =  vX.z;	M2.tab[4] =  0;		M2.tab[5] = -vX.x ;
	M2.tab[6] = -vX.y;	M2.tab[7] =  vX.x;	M2.tab[8] =  0;


	M = cos (angle) * I + (1- cos (angle)) * M1 + sin (angle) * M2;

	vecteur temp = vY;

	vY = M *  temp;
	vY.normalise();

	vZ = vX^vY;
	vZ.normalise();
}

void pitch(vecteur & vX,vecteur & vY, vecteur & vZ, float angle)
{
	matrice33 M,I,M1,M2;

	memcpy(M.tab,matrice33_0,9 * sizeof(float));
	memcpy(I.tab,matrice33_Id,9 * sizeof(float));

	M1.tab[0] = vZ.x * vZ.x;	M1.tab[1] = vZ.y * vZ.x;	M1.tab[2] = vZ.z * vZ.x;
	M1.tab[3] = vZ.x * vZ.y;	M1.tab[4] = vZ.y * vZ.y;	M1.tab[5] = vZ.z * vZ.y;
	M1.tab[6] = vZ.x * vZ.z;	M1.tab[7] = vZ.y * vZ.z;	M1.tab[8] = vZ.z * vZ.z;

	M2.tab[0] =  0;		M2.tab[1] = -vZ.z;	M2.tab[2] =  vZ.y;
	M2.tab[3] =  vZ.z;	M2.tab[4] =  0;		M2.tab[5] = -vZ.x ;
	M2.tab[6] = -vZ.y;	M2.tab[7] =  vZ.x;	M2.tab[8] =  0;


	M = cos (angle) * I + (1- cos (angle)) * M1 + sin (angle) * M2;

	vecteur temp = vX;
	vX = M *  temp;
	vX.normalise();

	vY = vZ^vX;
	vY.normalise();
}