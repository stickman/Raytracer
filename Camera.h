#ifndef __CAMERA_H
#define __CAMERA_H

#include "Def.h"

struct camera
{
	
	float focale; //yvoc?
	float znear;  //au plus pres
	float zfar;	  //au plus loin

	point position;

	vecteur vXAxis;
	vecteur vYAxis;
	vecteur vZAxis;

};


#endif
