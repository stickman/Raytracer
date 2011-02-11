#ifndef __QUATERNION__H
#define __QUATERNION__H


#include "Def.h"
#include "Matrix.h"

void YAW(vecteur & vX,vecteur & vY, vecteur & vZ, float angle);
void roll(vecteur & vX,vecteur & vY, vecteur & vZ, float angle);
void pitch(vecteur & vX,vecteur & vY, vecteur & vZ, float angle);

#endif