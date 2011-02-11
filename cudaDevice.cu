#include "cudaDevice.h"

__constant__ int imgw = 640;
__constant__ int imgh = 480;

#define NUMDIM	3
#define RIGHT	0
#define LEFT	1
#define MIDDLE	2

__device__
bool HitBoundingBox(point min,point max,point orig,vecteur vdir,point & pCoord)
{
	float minB[3]; minB[0]=min.x; minB[1]=min.y; minB[2]=min.z;		//box
	float maxB[3]; maxB[0]=max.x; maxB[1]=max.y; maxB[2]=max.z;		//box 

	float origin[3]; origin[0]=orig.x; origin[1]=orig.y; origin[2]=orig.z;	

	float dir[3];	dir[0]=vdir.x; dir[1]=vdir.y; dir[2]=vdir.z;

	float coord[3];				

	bool inside = true;
	char quadrant[3];
	register int i;
	int whichPlane;
	float  maxT[3];
	float candidatePlane[3];

	/* Find candidate planes; this loop can be avoided if
   	rays cast all from the eye(assume perpsective view) */
	for (i=0; i<3; i++)
		if(origin[i] < minB[i])
		{
			quadrant[i] = LEFT;
			candidatePlane[i] = minB[i];
			inside = false;
		}
		else 
			if (origin[i] > maxB[i])
			{
			quadrant[i] = RIGHT;
			candidatePlane[i] = maxB[i];
			inside = false;
			}
			else	
			{
			quadrant[i] = MIDDLE;
			}

	/* Ray origin inside bounding box */
	if(inside)	
	{
		pCoord = orig;
		return true;
	}


	/* Calculate T distances to candidate planes */
	for (i = 0; i < 3; i++)
		if (quadrant[i] != MIDDLE && dir[i] !=0.)
			maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
		else
			maxT[i] = -1.;

	/* Get largest of the maxT's for final choice of intersection */
	whichPlane = 0;
	for (i = 1; i < 3; i++)
		if (maxT[whichPlane] < maxT[i])
			whichPlane = i;

	/* Check final candidate actually inside box */
	if (maxT[whichPlane] < 0.) return false;
	for (i = 0; i < 3; i++)
		if (whichPlane != i)
		{
			coord[i] = origin[i] + maxT[whichPlane] *dir[i];
			if (coord[i] < minB[i] || coord[i] > maxB[i])
				return false;
		}
		else 
		{
			coord[i] = candidatePlane[i];
		}

	pCoord.x = coord[0];
	pCoord.y = coord[1];
	pCoord.z = coord[2];
	return true;				/* ray hits box */
}	





__global__ void
cudaProcess(float* g_odata,	point		positionCam,
							vecteur vXAxisMG,	vecteur vYAxisMG,vecteur vZAxisMG,
							material	vectorMat[], int nbMat,
						    sphere    vectorSphere[], int nbSphere, 
						    light	vectorLight[],  int nbLights, cubemap cm,
							nodeCuda * rootNode,triangle * pMemTriangles)
{

	int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;
	int BaseIndex = 0;

	//Mise en memoire partagee des vecteurs de la cam
	__shared__ vecteur vXAxis[32];
	__shared__ vecteur vYAxis[32];
	__shared__ vecteur vZAxis[32];

	const vecteur vXAxistp = vXAxisMG;
	const vecteur vYAxistp = vYAxisMG;
	const vecteur vZAxistp = vZAxisMG;


	vXAxis[BaseIndex + threadIdx.x] = vXAxistp;
	vYAxis[BaseIndex + threadIdx.x] = vYAxistp;
	vZAxis[BaseIndex + threadIdx.x] = vZAxistp;

	__syncthreads();


	color output = {0.0f, 0.0f, 0.0f};

	float _X = x - imgh/2.0;
	float _Y = y - imgw/2.0;

	vecteur dir;

	dir.x = vXAxis[BaseIndex + threadIdx.x].x * _X + vYAxis[BaseIndex + threadIdx.x].x * _Y + vZAxis[BaseIndex + threadIdx.x].x * 500; //Coordonnée X
	dir.y = vXAxis[BaseIndex + threadIdx.x].y * _X + vYAxis[BaseIndex + threadIdx.x].y * _Y + vZAxis[BaseIndex + threadIdx.x].y * 500; //Coordonnée Y
	dir.z = vXAxis[BaseIndex + threadIdx.x].z * _X + vYAxis[BaseIndex + threadIdx.x].z * _Y + vZAxis[BaseIndex + threadIdx.x].z * 500; //Coordonnée Z

	//Normalise a vector
	float l = sqrtf( dir.x * dir.x + dir.y * dir.y + dir.z * dir.z );
	if (l != 0)
	{
		dir.x /= l;
		dir.y /= l;
		dir.z /= l;
	}

	ray viewRay = { positionCam,dir};
	viewRay.indice = 1.0;

	cudaAddRay (viewRay, vectorMat,nbMat, vectorSphere,nbSphere,  vectorLight,nbLights,cm,rootNode,pMemTriangles, &output);

	// gamma correction
	float invgamma = 0.45f; // c'est la valeur fixée par le standard sRGB

	g_odata[(x + imgw*y) * 4]	  = powf(output.red,  invgamma);
	g_odata[(x + imgw*y) * 4 + 1] = powf(output.green, invgamma);
	g_odata[(x + imgw*y) * 4 + 2] = powf(output.blue,  invgamma);
	g_odata[(x + imgw*y) * 4 + 3] = 0.0;
}

__device__ void normalise(vecteur *dir)
{
	float l = sqrtf( dir->x * dir->x + dir->y * dir->y + dir->z * dir->z );

	if (l != 0)
	{
		dir->x /= l;
		dir->y /= l;
		dir->z /= l;
	}
}

__device__ float lenght(vecteur v)
{
	return sqrtf( v.x * v.x + v.y * v.y + v.z * v.z );
}

__device__ bool hitSphere(const ray &r, const sphere &s, float &t)
{
	vecteur dist = s.pos - r.start; 
	if(lenght(dist)-0.001 <= s.size)
		return false; //On est dans la sphere...
	float B = r.dir * dist;
	float D = B*B - dist * dist + s.size * s.size; 
	if (D < 0.0f) 
		return false; 
	float t0 = B - sqrtf(D); 
	float t1 = B + sqrtf(D);
	bool retvalue = false;  
	if ((t0 > 0.1f) && (t0 < t)) 
	{
		t = t0;
		retvalue = true; 
	} 
	if ((t1 > 0.1f) && (t1 < t)) 
	{
		t = t1; 
		retvalue = true; 
	}
	return retvalue; 
}

__device__ bool intersect_RayTriangle(const ray &R, const triangle& T, float & distance, bool shadowCast = false )
{
    vecteur		u, v;           // triangle vectors
    vecteur		w0, w;          // ray vectors
    float		r, a, b;        // params to calc ray-plane intersect
	point		I;

    // get triangle edge vectors and plane normal
	u = T.B - T.A;
    v = T.C - T.A;

	w0 = R.start - T.A;
    a = -1.0 * (T.n*w0);
    b = T.n * R.dir;

	if (fabs(b) <= 0.0001) return false; //parralele ou dans le plan 

	if ( b > 0 &&  !shadowCast ) return false; // mauvaise face du triangle

    // get intersect point of ray with triangle plane
    r = a / b;
	if (r < 0.1f || r >= distance)       // ray goes away from triangle or already intersect!
        return false;

	I = R.start + r * R.dir;           // intersect point of ray and plane

    // is I inside T?
    float    uu, uv, vv, wu, wv, D;
    uu = u*u;
    uv = u*v;
    vv = v*v;
	w = I - T.A;
    wu = w*u;
    wv = w*v;
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)        // I is outside T
        return false;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return false;

	distance = r;
    return true;                      // I is in T
}

__device__ void cudaAddRay( ray						viewRay, 
							material	materials[], int nbMat,
						    sphere    spheres[], int nbSphere, 
						    light		lights[], int nbLights,
							cubemap cm,
							nodeCuda * rootNode,triangle * pMemTriangles,
						    color*	output)
{
	viewRay.indice = 1;
	output->red   = 1.0f;
	output->green = 1.0f;
	output->blue  = 1.0f;

	float coefReflect = 1.0f;
	float coefRefract = 1.0f;

	int level = 0;
	do
	{        
		int currentSphere=-1;
		int currentTriangle=-1;
        
	    float t = 20000.0f;

		for (int i = 0; i < nbSphere ; ++i)
	    {
		    if (hitSphere(viewRay, spheres[i], t))
		    {
			    currentSphere = i;
		    }
	    }

		/** Debut parcour BSP **/

		nodeCuda* actuel = rootNode;
		nodeCuda* precedent = NULL;
		nodeCuda* suivant =NULL;

		point out;
		   
		while(actuel)
		{
			if (precedent == actuel)//FINI!
			{
				break;				
			}
			if (precedent == actuel->parent)
			{
				precedent = actuel;
				if (actuel->gauche)
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							suivant   = actuel->gauche;
						}
						else
						{
							suivant = actuel->parent;
						}
					}
					else
					{
						suivant = actuel->parent;
					}
					
				}
				else
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{						
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t))
									{
										currentSphere = -1;
										currentTriangle = actuel->tabIdTriangle[i];
									}
								}
							}
						}
					}					
					suivant = actuel->parent;
				}
			}				
			if (precedent == actuel->gauche)//on visite a droite!
			{
				precedent = actuel;
				suivant   = actuel->droit;					
			}	
			if (precedent == actuel->droit)//on visite a gauche
			{
				precedent = actuel;
				suivant   = actuel->parent;					
			}
			actuel = suivant;
		}
	    if ( (currentSphere == -1) && (currentTriangle == -1) )
		{
			
			if (level ==0)
			{
				*output =  readCubemap(cm, viewRay) ;
				return;
			}

			*output += coefReflect * readCubemap(cm, viewRay) ;			
			break;
		}
		
	    point ptHitPoint = viewRay.start + t * viewRay.dir;
        
		vecteur vNormal;
		material currentMat;

		if ( currentSphere != -1)
		{
			vNormal = ptHitPoint - spheres[currentSphere].pos;
			float temp = vNormal * vNormal;
			if (temp == 0.0f)
				break;
			temp = 1.0f / sqrtf(temp);
			vNormal = temp * vNormal;
			currentMat = materials[spheres[currentSphere].materialId];
		}
		else
		{
			vNormal = (pMemTriangles+currentTriangle)->n;
			 //currentMat = materials[0];
			currentMat = materials[(pMemTriangles+currentTriangle)->materialId];
		}
		
		coefReflect *= currentMat.reflectivity;
		coefRefract *= currentMat.transparency;

		if (level ==0)
		{
			output->red   = currentMat.ambient.red * 0.1;
			output->green = currentMat.ambient.green * 0.1;
			output->blue  = currentMat.ambient.blue * 0.1;
		}


		ray lightRay;
		lightRay.start = ptHitPoint;

		if(currentMat.transparency != 0.0f)
		{
			float fViewProjection = viewRay.dir * vNormal;
			float fCosThetaI = fabsf(fViewProjection); 
			float fSinThetaI = sqrtf(1 - fCosThetaI * fCosThetaI);
			float fSinThetaT = (1.0 / 1.5) * fSinThetaI;
            float fCosThetaT = sqrtf(1 - fSinThetaT * fSinThetaT);

			// Transparence -> le rayon continue sa route et prend la couleur de transparence de l'objet.
			ray refractRay;
			refractRay.start = ptHitPoint;
			refractRay.dir = viewRay.dir + fCosThetaI * vNormal;

			refractRay.dir = (1.0 / 1.5) * refractRay.dir;
			refractRay.dir += (-fCosThetaT) * vNormal;
			normalise(&refractRay.dir);
			color temp;
			cudaCalcTransparencyRay(refractRay,materials,  nbMat,
												  spheres,  nbSphere, 
												  lights, nbLights,
												  cm,
												  rootNode,pMemTriangles,
												  &temp);

			output->red   += temp.red * (1-coefReflect) * coefRefract; //REFRACTION
			output->green += temp.green* (1-coefReflect) * coefRefract;
			output->blue  += temp.blue* (1-coefReflect) * coefRefract;
		}

		for (unsigned int j = 0; j < nbLights ; ++j)
		{
			light currentLight = lights[j];

		    lightRay.dir = currentLight.pos - ptHitPoint;

            float fLightProjection = lightRay.dir * vNormal;

			if ( fLightProjection <= 0.0f )
				continue;

			float lightDist = lightRay.dir * lightRay.dir;
            
            float temp = lightDist;
		    if ( temp == 0.0f )
			    continue;
            temp = 1.0/ sqrtf(temp);
		    lightRay.dir = temp * lightRay.dir;

            fLightProjection = temp * fLightProjection;            

			bool inShadow = false;
            
            float t = lightDist;

			for (unsigned int i = 0; i <nbSphere ; ++i)
		    {
			    if (hitSphere(lightRay, spheres[i], t))
			    {
				    inShadow = true;
				    break;
			    }
		    }

			/** Debut parcour de l'arbre pour les ombres**/
			while(actuel)
			{
				if (precedent == actuel)//FINI!
				{
					break;				
				}
				if (precedent == actuel->parent)
				{
					precedent = actuel;
					if (actuel->gauche)
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							suivant   = actuel->gauche;
						}
						else
						{
							if (actuel->parent)	suivant = actuel->parent;
						}
					}
					else
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t,true))
									{
										inShadow = true;
										break;
									}
								}
							}
						}
						else
						{
							suivant = actuel->parent;
						}
				
					}
				}				
				if (precedent == actuel->gauche)//on visite a droite!
				{
					precedent = actuel;
					suivant   = actuel->droit;					
				}	
				if (precedent == actuel->droit)//on visite a gauche
				{
					precedent = actuel;
					suivant   = actuel->parent;					
				}
				actuel = suivant;
			}

			if (!inShadow)
			{
				float lambert = (lightRay.dir * vNormal) * coefReflect * (1-coefRefract);
				output->red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
				output->green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
				output->blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;

				// Blinn

                float fViewProjection = viewRay.dir * vNormal;
				vecteur blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f )
				{
					float blinn = 1.0/sqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                    blinn = coefReflect * (1-coefRefract) * powf(blinn, currentMat.shininess);
				
					output->red += blinn *currentMat.specular.red  * currentLight.intensity.red;
					output->green += blinn *currentMat.specular.green  * currentLight.intensity.green;
					output->blue += blinn *currentMat.specular.blue  * currentLight.intensity.blue;
				}
			}
		}

		float reflet = 2.0f * (viewRay.dir * vNormal);
        viewRay.start = ptHitPoint;
		viewRay.dir = viewRay.dir - reflet * vNormal;
		normalise(&viewRay.dir);
		level++;
	}
	while ((coefReflect > 0.0f || coefRefract >0.0f) && (level < 10));  
    //return output;
}


__device__ void cudaCalcTransparencyRay(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
						    color*	output)
							{
	viewRay.indice = 1;
	output->red   = 1.0f;
	output->green = 1.0f;
	output->blue  = 1.0f;

	float coefReflect = 1.0f;
	float coefRefract = 1.0f;

	int level = 0;
	do
	{        
		int currentSphere=-1;
		int currentTriangle=-1;
        
	    float t = 20000.0f;

		for (int i = 0; i < nbSphere ; ++i)
	    {
		    if (hitSphere(viewRay, spheres[i], t))
		    {
			    currentSphere = i;
		    }
	    }

		/** Debut parcour BSP **/

		nodeCuda* actuel = rootNode;
		nodeCuda* precedent = NULL;
		nodeCuda* suivant =NULL;

		point out;
		   
		while(actuel)
		{
			if (precedent == actuel)//FINI!
			{
				break;				
			}
			if (precedent == actuel->parent)
			{
				precedent = actuel;
				if (actuel->gauche)
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							suivant   = actuel->gauche;
						}
						else
						{
							suivant = actuel->parent;
						}
					}
					else
					{
						suivant = actuel->parent;
					}
					
				}
				else
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{						
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t))
									{
										currentSphere = -1;
										currentTriangle = actuel->tabIdTriangle[i];
									}
								}
							}
						}
					}					
					suivant = actuel->parent;
				}
			}				
			if (precedent == actuel->gauche)//on visite a droite!
			{
				precedent = actuel;
				suivant   = actuel->droit;					
			}	
			if (precedent == actuel->droit)//on visite a gauche
			{
				precedent = actuel;
				suivant   = actuel->parent;					
			}
			actuel = suivant;
		}
	    if ( (currentSphere == -1) && (currentTriangle == -1) )
		{
			
			if (level ==0)
			{
				*output =  readCubemap(cm, viewRay) ;
				return;
			}

			*output += coefReflect * readCubemap(cm, viewRay) ;			
			break;
		}
		
	    point ptHitPoint = viewRay.start + t * viewRay.dir;
        
		vecteur vNormal;
		material currentMat;

		if ( currentSphere != -1)
		{
			vNormal = ptHitPoint - spheres[currentSphere].pos;
			float temp = vNormal * vNormal;
			if (temp == 0.0f)
				break;
			temp = 1.0f / sqrtf(temp);
			vNormal = temp * vNormal;
			currentMat = materials[spheres[currentSphere].materialId];
		}
		else
		{
			vNormal = (pMemTriangles+currentTriangle)->n;
			//currentMat = materials[0];
			currentMat = materials[(pMemTriangles+currentTriangle)->materialId];
		}
		
		coefReflect *= currentMat.reflectivity;
		coefRefract *= currentMat.transparency;

		if (level ==0)
		{
			output->red   = currentMat.ambient.red * 0.1;
			output->green = currentMat.ambient.green * 0.1;
			output->blue  = currentMat.ambient.blue * 0.1;
		}


		ray lightRay;
		lightRay.start = ptHitPoint;

		if(currentMat.transparency != 0.0f)
		{
			float fViewProjection = viewRay.dir * vNormal;
			float fCosThetaI = fabsf(fViewProjection); 
			float fSinThetaI = sqrtf(1 - fCosThetaI * fCosThetaI);
			float fSinThetaT = (1.0 / 1.5) * fSinThetaI;
            float fCosThetaT = sqrtf(1 - fSinThetaT * fSinThetaT);

			// Transparence -> le rayon continue sa route et prend la couleur de transparence de l'objet.
			ray refractRay;
			refractRay.start = ptHitPoint;
			refractRay.dir = viewRay.dir + fCosThetaI * vNormal;

			refractRay.dir = (1.0 / 1.5) * refractRay.dir;
			refractRay.dir += (-fCosThetaT) * vNormal;
			normalise(&refractRay.dir);
			color temp;
			cudaCalcTransparencyRay2(refractRay,materials,  nbMat,
												  spheres,  nbSphere, 
												  lights, nbLights,
												  cm,
												  rootNode,pMemTriangles,
												  &temp);

			output->red   += temp.red * (1-coefReflect) * coefRefract;;
			output->green += temp.green * (1-coefReflect) * coefRefract;;
			output->blue  += temp.blue * (1-coefReflect) * coefRefract;;
		}

		for (unsigned int j = 0; j < nbLights ; ++j)
		{
			light currentLight = lights[j];

		    lightRay.dir = currentLight.pos - ptHitPoint;

            float fLightProjection = lightRay.dir * vNormal;

			if ( fLightProjection <= 0.0f )
				continue;

			float lightDist = lightRay.dir * lightRay.dir;
            
            float temp = lightDist;
		    if ( temp == 0.0f )
			    continue;
            temp = 1.0/ sqrtf(temp);
		    lightRay.dir = temp * lightRay.dir;

            fLightProjection = temp * fLightProjection;            

			bool inShadow = false;
            
            float t = lightDist;

			for (unsigned int i = 0; i <nbSphere ; ++i)
		    {
			    if (hitSphere(lightRay, spheres[i], t))
			    {
				    inShadow = true;
				    break;
			    }
		    }

			/** Debut parcour de l'arbre pour les ombres**/
			while(actuel)
			{
				if (precedent == actuel)//FINI!
				{
					break;				
				}
				if (precedent == actuel->parent)
				{
					precedent = actuel;
					if (actuel->gauche)
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							suivant   = actuel->gauche;
						}
						else
						{
							if (actuel->parent)	suivant = actuel->parent;
						}
					}
					else
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t,true))
									{
										inShadow = true;
										break;
									}
								}
							}
						}
						else
						{
							suivant = actuel->parent;
						}
				
					}
				}				
				if (precedent == actuel->gauche)//on visite a droite!
				{
					precedent = actuel;
					suivant   = actuel->droit;					
				}	
				if (precedent == actuel->droit)//on visite a gauche
				{
					precedent = actuel;
					suivant   = actuel->parent;					
				}
				actuel = suivant;
			}

			if (!inShadow)
			{
				float lambert = (lightRay.dir * vNormal) * coefReflect * (1-coefRefract);
				output->red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
				output->green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
				output->blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;

				// Blinn 
                // La direction de Blinn est exactement à mi chemin entre le rayon
                // lumineux et le rayon de vue. 
                // On calcule le vecteur de Blinn et on le rend unitaire
                // puis on calcule le coéfficient de blinn
                // qui est la contribution spéculaire de la lumière courante.

                float fViewProjection = viewRay.dir * vNormal;
				vecteur blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f )
				{
					float blinn = 1.0/sqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                    blinn = coefReflect * (1-coefRefract) * powf(blinn, currentMat.shininess);
				
					output->red += blinn *currentMat.specular.red  * currentLight.intensity.red;
					output->green += blinn *currentMat.specular.green  * currentLight.intensity.green;
					output->blue += blinn *currentMat.specular.blue  * currentLight.intensity.blue;
				}
			}
		}

		float reflet = 2.0f * (viewRay.dir * vNormal);
        viewRay.start = ptHitPoint;
		viewRay.dir = viewRay.dir - reflet * vNormal;
		normalise(&viewRay.dir);
		level++;
	}
	while ((coefReflect > 0.0f || coefRefract >0.0f) && (level < 10));  
    //return output;
}
__device__ void cudaCalcTransparencyRay2(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
						    color*	output)
							{
	viewRay.indice = 1;
	output->red   = 1.0f;
	output->green = 1.0f;
	output->blue  = 1.0f;

	float coefReflect = 1.0f;
	float coefRefract = 1.0f;

	int level = 0;
	do
	{        
		int currentSphere=-1;
		int currentTriangle=-1;
        
	    float t = 20000.0f;

		for (int i = 0; i < nbSphere ; ++i)
	    {
		    if (hitSphere(viewRay, spheres[i], t))
		    {
			    currentSphere = i;
		    }
	    }

		/** Debut parcour BSP **/

		nodeCuda* actuel = rootNode;
		nodeCuda* precedent = NULL;
		nodeCuda* suivant =NULL;

		point out;
		   
		while(actuel)
		{
			if (precedent == actuel)//FINI!
			{
				break;				
			}
			if (precedent == actuel->parent)
			{
				precedent = actuel;
				if (actuel->gauche)
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							suivant   = actuel->gauche;
						}
						else
						{
							suivant = actuel->parent;
						}
					}
					else
					{
						suivant = actuel->parent;
					}
					
				}
				else
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{						
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t))
									{
										currentSphere = -1;
										currentTriangle = actuel->tabIdTriangle[i];
									}
								}
							}
						}
					}					
					suivant = actuel->parent;
				}
			}				
			if (precedent == actuel->gauche)//on visite a droite!
			{
				precedent = actuel;
				suivant   = actuel->droit;					
			}	
			if (precedent == actuel->droit)//on visite a gauche
			{
				precedent = actuel;
				suivant   = actuel->parent;					
			}
			actuel = suivant;
		}
	    if ( (currentSphere == -1) && (currentTriangle == -1) )
		{
			
			if (level ==0)
			{
				*output =  readCubemap(cm, viewRay) ;
				return;
			}

			*output += coefReflect * readCubemap(cm, viewRay) ;			
			break;
		}
		
	    point ptHitPoint = viewRay.start + t * viewRay.dir;
        
		vecteur vNormal;
		material currentMat;

		if ( currentSphere != -1)
		{
			vNormal = ptHitPoint - spheres[currentSphere].pos;
			float temp = vNormal * vNormal;
			if (temp == 0.0f)
				break;
			temp = 1.0f / sqrtf(temp);
			vNormal = temp * vNormal;
			currentMat = materials[spheres[currentSphere].materialId];
		}
		else
		{
			vNormal = (pMemTriangles+currentTriangle)->n;
			 //currentMat = materials[0];
			currentMat = materials[(pMemTriangles+currentTriangle)->materialId];
		}
		
		coefReflect *= currentMat.reflectivity;
		coefRefract *= currentMat.transparency;

		if (level ==0)
		{
			output->red   = currentMat.ambient.red * 0.1;
			output->green = currentMat.ambient.green * 0.1;
			output->blue  = currentMat.ambient.blue * 0.1;
		}


		ray lightRay;
		lightRay.start = ptHitPoint;

		if(currentMat.transparency != 0.0f)
		{
			float fViewProjection = viewRay.dir * vNormal;
			float fCosThetaI = fabsf(fViewProjection); 
			float fSinThetaI = sqrtf(1 - fCosThetaI * fCosThetaI);
			float fSinThetaT = (1.0 / 1.5) * fSinThetaI;
            float fCosThetaT = sqrtf(1 - fSinThetaT * fSinThetaT);

			// Transparence -> le rayon continue sa route et prend la couleur de transparence de l'objet.
			ray refractRay;
			refractRay.start = ptHitPoint;
			refractRay.dir = viewRay.dir + fCosThetaI * vNormal;

			refractRay.dir = (1.0 / 1.5) * refractRay.dir;
			refractRay.dir += (-fCosThetaT) * vNormal;
			normalise(&refractRay.dir);
			color temp;
			cudaCalcTransparencyRay3(refractRay,materials,  nbMat,
												  spheres,  nbSphere, 
												  lights, nbLights,
												  cm,
												  rootNode,pMemTriangles,
												  &temp);

			output->red   += temp.red * (1-coefReflect) * coefRefract;;
			output->green += temp.green * (1-coefReflect) * coefRefract;;
			output->blue  += temp.blue * (1-coefReflect) * coefRefract;;
		}

		for (unsigned int j = 0; j < nbLights ; ++j)
		{
			light currentLight = lights[j];

		    lightRay.dir = currentLight.pos - ptHitPoint;

            float fLightProjection = lightRay.dir * vNormal;

			if ( fLightProjection <= 0.0f )
				continue;

			float lightDist = lightRay.dir * lightRay.dir;
            
            float temp = lightDist;
		    if ( temp == 0.0f )
			    continue;
            temp = 1.0/ sqrtf(temp);
		    lightRay.dir = temp * lightRay.dir;

            fLightProjection = temp * fLightProjection;            

			bool inShadow = false;
            
            float t = lightDist;

			for (unsigned int i = 0; i <nbSphere ; ++i)
		    {
			    if (hitSphere(lightRay, spheres[i], t))
			    {
				    inShadow = true;
				    break;
			    }
		    }

			/** Debut parcour de l'arbre pour les ombres**/
			while(actuel)
			{
				if (precedent == actuel)//FINI!
				{
					break;				
				}
				if (precedent == actuel->parent)
				{
					precedent = actuel;
					if (actuel->gauche)
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							suivant   = actuel->gauche;
						}
						else
						{
							if (actuel->parent)	suivant = actuel->parent;
						}
					}
					else
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t,true))
									{
										inShadow = true;
										break;
									}
								}
							}
						}
						else
						{
							suivant = actuel->parent;
						}
				
					}
				}				
				if (precedent == actuel->gauche)//on visite a droite!
				{
					precedent = actuel;
					suivant   = actuel->droit;					
				}	
				if (precedent == actuel->droit)//on visite a gauche
				{
					precedent = actuel;
					suivant   = actuel->parent;					
				}
				actuel = suivant;
			}

			if (!inShadow)
			{
				float lambert = (lightRay.dir * vNormal) * coefReflect * (1-coefRefract);
				output->red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
				output->green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
				output->blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;

				// Blinn 
                // La direction de Blinn est exactement à mi chemin entre le rayon
                // lumineux et le rayon de vue. 
                // On calcule le vecteur de Blinn et on le rend unitaire
                // puis on calcule le coéfficient de blinn
                // qui est la contribution spéculaire de la lumière courante.

                float fViewProjection = viewRay.dir * vNormal;
				vecteur blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f )
				{
					float blinn = 1.0/sqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                    blinn = coefReflect * (1-coefRefract) * powf(blinn, currentMat.shininess);
				
					output->red += blinn *currentMat.specular.red  * currentLight.intensity.red;
					output->green += blinn *currentMat.specular.green  * currentLight.intensity.green;
					output->blue += blinn *currentMat.specular.blue  * currentLight.intensity.blue;
				}
			}
		}

		float reflet = 2.0f * (viewRay.dir * vNormal);
        viewRay.start = ptHitPoint;
		viewRay.dir = viewRay.dir - reflet * vNormal;
		normalise(&viewRay.dir);
		level++;
	}
	while ((coefReflect > 0.0f || coefRefract >0.0f) && (level < 8));  
    //return output;
}
__device__ void cudaCalcTransparencyRay3(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
						    color*	output)
							{
	viewRay.indice = 1;
	output->red   = 1.0f;
	output->green = 1.0f;
	output->blue  = 1.0f;

	float coefReflect = 1.0f;
	float coefRefract = 1.0f;

	int level = 0;
	do
	{        
		int currentSphere=-1;
		int currentTriangle=-1;
        
	    float t = 20000.0f;

		for (int i = 0; i < nbSphere ; ++i)
	    {
		    if (hitSphere(viewRay, spheres[i], t))
		    {
			    currentSphere = i;
		    }
	    }

		/** Debut parcour BSP **/

		nodeCuda* actuel = rootNode;
		nodeCuda* precedent = NULL;
		nodeCuda* suivant =NULL;

		point out;
		   
		while(actuel)
		{
			if (precedent == actuel)//FINI!
			{
				break;				
			}
			if (precedent == actuel->parent)
			{
				precedent = actuel;
				if (actuel->gauche)
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							suivant   = actuel->gauche;
						}
						else
						{
							suivant = actuel->parent;
						}
					}
					else
					{
						suivant = actuel->parent;
					}
					
				}
				else
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{						
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t))
									{
										currentSphere = -1;
										currentTriangle = actuel->tabIdTriangle[i];
									}
								}
							}
						}
					}					
					suivant = actuel->parent;
				}
			}				
			if (precedent == actuel->gauche)//on visite a droite!
			{
				precedent = actuel;
				suivant   = actuel->droit;					
			}	
			if (precedent == actuel->droit)//on visite a gauche
			{
				precedent = actuel;
				suivant   = actuel->parent;					
			}
			actuel = suivant;
		}
	    if ( (currentSphere == -1) && (currentTriangle == -1) )
		{
			
			if (level ==0)
			{
				*output =  readCubemap(cm, viewRay) ;
				return;
			}

			*output += coefReflect * readCubemap(cm, viewRay) ;			
			break;
		}
		
	    point ptHitPoint = viewRay.start + t * viewRay.dir;
        
		vecteur vNormal;
		material currentMat;

		if ( currentSphere != -1)
		{
			vNormal = ptHitPoint - spheres[currentSphere].pos;
			float temp = vNormal * vNormal;
			if (temp == 0.0f)
				break;
			temp = 1.0f / sqrtf(temp);
			vNormal = temp * vNormal;
			currentMat = materials[spheres[currentSphere].materialId];
		}
		else
		{
			vNormal = (pMemTriangles+currentTriangle)->n;
			 //currentMat = materials[0];
			currentMat = materials[(pMemTriangles+currentTriangle)->materialId];
		}
		
		coefReflect *= currentMat.reflectivity;
		coefRefract *= currentMat.transparency;

		if (level ==0)
		{
			output->red   = currentMat.ambient.red * 0.1;
			output->green = currentMat.ambient.green * 0.1;
			output->blue  = currentMat.ambient.blue * 0.1;
		}


		ray lightRay;
		lightRay.start = ptHitPoint;

		if(currentMat.transparency != 0.0f)
		{
			float fViewProjection = viewRay.dir * vNormal;
			float fCosThetaI = fabsf(fViewProjection); 
			float fSinThetaI = sqrtf(1 - fCosThetaI * fCosThetaI);
			float fSinThetaT = (1.0 / 1.5) * fSinThetaI;
            float fCosThetaT = sqrtf(1 - fSinThetaT * fSinThetaT);

			// Transparence -> le rayon continue sa route et prend la couleur de transparence de l'objet.
			ray refractRay;
			refractRay.start = ptHitPoint;
			refractRay.dir = viewRay.dir + fCosThetaI * vNormal;

			refractRay.dir = (1.0 / 1.5) * refractRay.dir;
			refractRay.dir += (-fCosThetaT) * vNormal;
			normalise(&refractRay.dir);
			color temp;
			cudaCalcTransparencyRay4(refractRay,materials,  nbMat,
												  spheres,  nbSphere, 
												  lights, nbLights,
												  cm,
												  rootNode,pMemTriangles,
												  &temp);

			output->red   += temp.red * (1-coefReflect) * coefRefract;;
			output->green += temp.green * (1-coefReflect) * coefRefract;;
			output->blue  += temp.blue * (1-coefReflect) * coefRefract;;
		}

		for (unsigned int j = 0; j < nbLights ; ++j)
		{
			light currentLight = lights[j];

		    lightRay.dir = currentLight.pos - ptHitPoint;

            float fLightProjection = lightRay.dir * vNormal;

			if ( fLightProjection <= 0.0f )
				continue;

			float lightDist = lightRay.dir * lightRay.dir;
            
            float temp = lightDist;
		    if ( temp == 0.0f )
			    continue;
            temp = 1.0/ sqrtf(temp);
		    lightRay.dir = temp * lightRay.dir;

            fLightProjection = temp * fLightProjection;            

			bool inShadow = false;
            
            float t = lightDist;

			for (unsigned int i = 0; i <nbSphere ; ++i)
		    {
			    if (hitSphere(lightRay, spheres[i], t))
			    {
				    inShadow = true;
				    break;
			    }
		    }

			/** Debut parcour de l'arbre pour les ombres**/
			while(actuel)
			{
				if (precedent == actuel)//FINI!
				{
					break;				
				}
				if (precedent == actuel->parent)
				{
					precedent = actuel;
					if (actuel->gauche)
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							suivant   = actuel->gauche;
						}
						else
						{
							if (actuel->parent)	suivant = actuel->parent;
						}
					}
					else
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t,true))
									{
										inShadow = true;
										break;
									}
								}
							}
						}
						else
						{
							suivant = actuel->parent;
						}
				
					}
				}				
				if (precedent == actuel->gauche)//on visite a droite!
				{
					precedent = actuel;
					suivant   = actuel->droit;					
				}	
				if (precedent == actuel->droit)//on visite a gauche
				{
					precedent = actuel;
					suivant   = actuel->parent;					
				}
				actuel = suivant;
			}

			if (!inShadow)
			{
				float lambert = (lightRay.dir * vNormal) * coefReflect * (1-coefRefract);
				output->red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
				output->green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
				output->blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;

				// Blinn 
                // La direction de Blinn est exactement à mi chemin entre le rayon
                // lumineux et le rayon de vue. 
                // On calcule le vecteur de Blinn et on le rend unitaire
                // puis on calcule le coéfficient de blinn
                // qui est la contribution spéculaire de la lumière courante.

                float fViewProjection = viewRay.dir * vNormal;
				vecteur blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f )
				{
					float blinn = 1.0/sqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                    blinn = coefReflect * (1-coefRefract) * powf(blinn, currentMat.shininess);
				
					output->red += blinn *currentMat.specular.red  * currentLight.intensity.red;
					output->green += blinn *currentMat.specular.green  * currentLight.intensity.green;
					output->blue += blinn *currentMat.specular.blue  * currentLight.intensity.blue;
				}
			}
		}

		float reflet = 2.0f * (viewRay.dir * vNormal);
        viewRay.start = ptHitPoint;
		viewRay.dir = viewRay.dir - reflet * vNormal;
		normalise(&viewRay.dir);
		level++;
	}
	while ((coefReflect > 0.0f || coefRefract >0.0f) && (level < 6));  
    //return output;
}
__device__ void cudaCalcTransparencyRay4(ray viewRay, 
							material	materials[],
							int nbMat,
						    sphere    spheres[],
							int nbSphere, 
						    light	lights[],
							int nbLights,
							cubemap cm,
							nodeCuda* rootNode,triangle * pMemTriangles,
						    color*	output)
{
	viewRay.indice = 1;
	output->red   = 1.0f;
	output->green = 1.0f;
	output->blue  = 1.0f;

	float coefReflect = 1.0f;
	float coefRefract = 1.0f;

	int level = 0;
	do
	{        
		int currentSphere=-1;
		int currentTriangle=-1;
        
	    float t = 20000.0f;

		for (int i = 0; i < nbSphere ; ++i)
	    {
		    if (hitSphere(viewRay, spheres[i], t))
		    {
			    currentSphere = i;
		    }
	    }

		/** Debut parcour BSP **/

		nodeCuda* actuel = rootNode;
		nodeCuda* precedent = NULL;
		nodeCuda* suivant =NULL;

		point out;
		   
		while(actuel)
		{
			if (precedent == actuel)//FINI!
			{
				break;				
			}
			if (precedent == actuel->parent)
			{
				precedent = actuel;
				if (actuel->gauche)
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							suivant   = actuel->gauche;
						}
						else
						{
							suivant = actuel->parent;
						}
					}
					else
					{
						suivant = actuel->parent;
					}
					
				}
				else
				{
					if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
					{						
						vecteur v = out-viewRay.start;
						if (lenght(v) <= t )
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t))
									{
										currentSphere = -1;
										currentTriangle = actuel->tabIdTriangle[i];
									}
								}
							}
						}
					}					
					suivant = actuel->parent;
				}
			}				
			if (precedent == actuel->gauche)//on visite a droite!
			{
				precedent = actuel;
				suivant   = actuel->droit;					
			}	
			if (precedent == actuel->droit)//on visite a gauche
			{
				precedent = actuel;
				suivant   = actuel->parent;					
			}
			actuel = suivant;
		}
	    if ( (currentSphere == -1) && (currentTriangle == -1) )
		{
			
			if (level ==0)
			{
				*output =  readCubemap(cm, viewRay) ;
				return;
			}

			*output += coefReflect * readCubemap(cm, viewRay) ;			
			break;
		}
		
	    point ptHitPoint = viewRay.start + t * viewRay.dir;
        
		vecteur vNormal;
		material currentMat;

		if ( currentSphere != -1)
		{
			vNormal = ptHitPoint - spheres[currentSphere].pos;
			float temp = vNormal * vNormal;
			if (temp == 0.0f)
				break;
			temp = 1.0f / sqrtf(temp);
			vNormal = temp * vNormal;
			currentMat = materials[spheres[currentSphere].materialId];
		}
		else
		{
			vNormal = (pMemTriangles+currentTriangle)->n;
			//currentMat = materials[0];
			currentMat = materials[(pMemTriangles+currentTriangle)->materialId];
		}
		
		coefReflect *= currentMat.reflectivity;
		coefRefract *= currentMat.transparency;

		if (level ==0)
		{
			output->red   = currentMat.ambient.red * 0.1;
			output->green = currentMat.ambient.green * 0.1;
			output->blue  = currentMat.ambient.blue * 0.1;
		}


		ray lightRay;
		lightRay.start = ptHitPoint;

		for (unsigned int j = 0; j < nbLights ; ++j)
		{
			light currentLight = lights[j];

		    lightRay.dir = currentLight.pos - ptHitPoint;

            float fLightProjection = lightRay.dir * vNormal;

			if ( fLightProjection <= 0.0f )
				continue;

			float lightDist = lightRay.dir * lightRay.dir;
            
            float temp = lightDist;
		    if ( temp == 0.0f )
			    continue;
            temp = 1.0/ sqrtf(temp);
		    lightRay.dir = temp * lightRay.dir;

            fLightProjection = temp * fLightProjection;            

			bool inShadow = false;
            
            float t = lightDist;

			for (unsigned int i = 0; i <nbSphere ; ++i)
		    {
			    if (hitSphere(lightRay, spheres[i], t))
			    {
				    inShadow = true;
				    break;
			    }
		    }

			/** Debut parcour de l'arbre pour les ombres**/
			while(actuel)
			{
				if (precedent == actuel)//FINI!
				{
					break;				
				}
				if (precedent == actuel->parent)
				{
					precedent = actuel;
					if (actuel->gauche)
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							suivant   = actuel->gauche;
						}
						else
						{
							if (actuel->parent)	suivant = actuel->parent;
						}
					}
					else
					{
						if (HitBoundingBox(actuel->min,actuel->max,viewRay.start,viewRay.dir,out))
						{
							for (int i = 0; i < 16 ; ++i)
							{
								if (actuel->tabIdTriangle[i] != -1)
								{
									if (intersect_RayTriangle(viewRay, *(pMemTriangles + actuel->tabIdTriangle[i] ), t,true))
									{
										inShadow = true;
										break;
									}
								}
							}
						}
						else
						{
							suivant = actuel->parent;
						}
				
					}
				}				
				if (precedent == actuel->gauche)//on visite a droite!
				{
					precedent = actuel;
					suivant   = actuel->droit;					
				}	
				if (precedent == actuel->droit)//on visite a gauche
				{
					precedent = actuel;
					suivant   = actuel->parent;					
				}
				actuel = suivant;
			}

			if (!inShadow)
			{
				float lambert = (lightRay.dir * vNormal) * coefReflect * (1-coefRefract);
				output->red += lambert * currentLight.intensity.red * currentMat.diffuse.red;
				output->green += lambert * currentLight.intensity.green * currentMat.diffuse.green;
				output->blue += lambert * currentLight.intensity.blue * currentMat.diffuse.blue;

				// Blinn 
                // La direction de Blinn est exactement à mi chemin entre le rayon
                // lumineux et le rayon de vue. 
                // On calcule le vecteur de Blinn et on le rend unitaire
                // puis on calcule le coéfficient de blinn
                // qui est la contribution spéculaire de la lumière courante.

                float fViewProjection = viewRay.dir * vNormal;
				vecteur blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f )
				{
					float blinn = 1.0/sqrtf(temp) * max(fLightProjection - fViewProjection , 0.0f);
                    blinn = coefReflect * (1-coefRefract) * powf(blinn, currentMat.shininess);
				
					output->red += blinn *currentMat.specular.red  * currentLight.intensity.red;
					output->green += blinn *currentMat.specular.green  * currentLight.intensity.green;
					output->blue += blinn *currentMat.specular.blue  * currentLight.intensity.blue;
				}
			}
		}

		float reflet = 2.0f * (viewRay.dir * vNormal);
        viewRay.start = ptHitPoint;
		viewRay.dir = viewRay.dir - reflet * vNormal;
		normalise(&viewRay.dir);
		level++;
	}
	while ((coefReflect > 0.0f || coefRefract >0.0f) && (level < 4));  
    //return output;
}

__device__
color readTexture(const color* tab, float u, float v, int sizeU, int sizeV)
{
    u = fabsf(u);
    v = fabsf(v);
    int umin = int(sizeU * u);
    int vmin = int(sizeV * v);
    int umax = int(sizeU * u) + 1;
    int vmax = int(sizeV * v) + 1;
    float ucoef = fabsf(sizeU * u - umin);
    float vcoef = fabsf(sizeV * v - vmin);
    
    // la texture est adresse sur [0,1]
    // le type d'adressage est la pour
    // determiner comment on gere les
    // coordonnées qui sont en dehors des bornes.
    // par défaut on ramène ce qui est plus
    // petit que zéro à zéro
    // et ce qui est plus grand que un à un.
    // (opérateur de saturation ou clamping)

    // Clamping est le défaut pour les textures
    umin = min(max(umin, 0), sizeU - 1);
    umax = min(max(umax, 0), sizeU - 1);
    vmin = min(max(vmin, 0), sizeV - 1);
    vmax = min(max(vmax, 0), sizeV - 1);

    // Ce qui suit est une interpolation bilinéaire
    // selon les deux coordonnées u et v.

    color output = 
			(1.0f - vcoef)	* ( (1.0f - ucoef)	* tab[umin  + sizeU * vmin]  +	ucoef	* tab[umax + sizeU * vmin] )
        +   vcoef			* ( (1.0f - ucoef)	* tab[umin  + sizeU * vmax]  +	ucoef	* tab[umax + sizeU * vmax] );
    return output;
}


__device__
color readCubemap(const cubemap & cm, ray myRay)
{
    color * currentColor ;
    color outputColor = {0.0f,0.0f,0.0f};
    if(!cm.textureCUDA)
    {
        return outputColor;
    }
    if ((fabsf(myRay.dir.x) >= fabsf(myRay.dir.y)) && (fabsf(myRay.dir.x) >= fabsf(myRay.dir.z)))
    {
        if (myRay.dir.x > 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::right * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.z / myRay.dir.x+ 1.0f) * 0.5f,  
                (myRay.dir.y / myRay.dir.x+ 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.x < 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::left * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.z / myRay.dir.x+ 1.0f) * 0.5f,
                1.0f - ( myRay.dir.y / myRay.dir.x + 1.0f) * 0.5f,  
                cm.sizeX, cm.sizeY);
        }
    }
    else if ((fabsf(myRay.dir.y) >= fabsf(myRay.dir.x)) && (fabsf(myRay.dir.y) >= fabsf(myRay.dir.z)))
    {
        if (myRay.dir.y > 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::up * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.y + 1.0f) * 0.5f,
                1.0f - (myRay.dir.z/ myRay.dir.y + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.y < 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::down * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                1.0f - (myRay.dir.x / myRay.dir.y + 1.0f) * 0.5f,  
                (myRay.dir.z/myRay.dir.y + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
    }
    else if ((fabsf(myRay.dir.z) >= fabsf(myRay.dir.x)) && (fabsf(myRay.dir.z) >= fabsf(myRay.dir.y)))
    {
        if (myRay.dir.z > 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::forward * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.z + 1.0f) * 0.5f,  
                (myRay.dir.y/myRay.dir.z + 1.0f) * 0.5f, cm.sizeX, cm.sizeY);
        }
        else if (myRay.dir.z < 0.0f)
        {
            currentColor = cm.textureCUDA + cubemap::backward * cm.sizeX * cm.sizeY;
            outputColor = readTexture(currentColor,  
                (myRay.dir.x / myRay.dir.z + 1.0f) * 0.5f,  
                1.0f - (myRay.dir.y /myRay.dir.z+1) * 0.5f, cm.sizeX, cm.sizeY);
        }
    }
    if (cm.bsRGB)
    {
       // On s'assure que les données sont ramenées dans un espace
       // de représentation linéaire au préalable
       outputColor.blue   = powf(outputColor.blue, 2.2f);
       outputColor.red    = powf(outputColor.red, 2.2f);
       outputColor.green  = powf(outputColor.green, 2.2f);
    }

    if (cm.bExposed)
    {
        // Les images LDR ont déjà été soumis à la fonction d'exposition
        // on les ramène à un espace similaire à l'espace de départ.
        outputColor.blue  = -logf(1.0f - outputColor.blue);
        outputColor.red   = -logf(1.0f - outputColor.red);
        outputColor.green = -logf(1.0f - outputColor.green);
    }

    outputColor.blue  /= cm.exposure;
    outputColor.red   /= cm.exposure;
    outputColor.green /= cm.exposure;

    return outputColor;
}


