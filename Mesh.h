#ifndef _MESH_H
#define _MESH_H

#include "triangle.h"

struct mesh
{
	char *				name;
	std::list<triangle>	triangles;
};


#endif