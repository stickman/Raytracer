#ifndef __DEF_H
#define __DEF_H

#include <string>
#include <cmath>
#include <list>

const double PIOVER180 = 0.017453292519943295769236907684886;
struct color 
{
    enum OFFSET 
    {
        OFFSET_RED = 0,
        OFFSET_GREEN = 1,
        OFFSET_BLUE = 2,
        OFFSET_MAX  = 3
    };

    float red, green, blue;

    inline color & operator += (const color &c2 )
	{
	    this->red +=  c2.red;
        this->green += c2.green;
        this->blue += c2.blue;
	    return *this;
    }

    inline float & getChannel(OFFSET offset )
    {
        return reinterpret_cast<float*>(this)[offset];
    }

    inline float getChannel(OFFSET offset ) const
    {
        return reinterpret_cast<const float*>(this)[offset];
    }
};

struct cubemap
{
	enum {
		up = 0,
		down = 1,
		right = 2,
		left = 3,
		forward = 4,
		backward = 5
	};

    int sizeX, sizeY;
	color* textureCUDA; 
    float exposure;
    bool bExposed;
    bool bsRGB;
    cubemap() : textureCUDA(0), exposure(1.0f), bExposed(false), bsRGB(false) {};
    bool Init(char * name[6]);
    void setExposure(float newExposure) {exposure = newExposure; }
};

struct vecteur
{	
	float x, y, z;

    vecteur& operator += (const vecteur &v2)
	{
	    x += v2.x;
        y += v2.y;
        z += v2.z;
	    return *this;
    }

    vecteur& operator -= (const vecteur &v2)
	{
	    x -= v2.x;
        y -= v2.y;
        z -= v2.z;
	    return *this;
    }

	vecteur& operator = (const vecteur & v2)
	{
		x = v2.x;
        y = v2.y;
        z = v2.z;
	    return *this;
	}

	inline float lenght ()
	{		
	    return  sqrtf( x * x + y * y + z * z );
    }

	vecteur& normalise ()
	{		
		float l = lenght();

		if (l != 0)
		{
			x /= l;
			y /= l;
			z /= l;
		}
	    return *this;
    }


};

struct point
{
	float x, y, z;

	point& operator += (const vecteur &v)
	{
		x += v.x;
        y += v.y;
        z += v.z;
	    return *this;
    }

	point& operator -= (const vecteur &v)
	{
	    x -= v.x;
        y -= v.y;
        z -= v.z;
	    return *this;
    }
};


struct material 
{	
	/*Parametres Phong*/
    float shininess; // >=1, plus c'est grand plus ca brille? (on peut monter très haut, 1000+...)   
	color ambient; //Couleur diffusée si une lumière ambiante est définie.
    color diffuse; //Couleur diffusée selon l'intensité des sources lumineuse atteignant le materiel.
    color specular; //Couleur d'une source luminese reflettée.
	
	/*Parametres utiles au ray tracing temps réel*/
	float reflectivity;//Coefficient de reflexion
	float densite;
	/*Autres parametres*/
	//color emission;//Si le materiel emet de la lumiere! Néon, ou autre truc phosphorescents

	//color reflective; //Couleur ajouté aux reflets	
	//color transparent; //Couleur donnée avec la transparence

	float transparency; //pourcentage de transparence
};
struct sphere 
{
	point pos;
	float size;
	int materialId;
};

struct light
{
	point pos;
	color intensity;
};

struct ray
{
	point start;
	vecteur dir;
	float indice;
};

inline point operator + (const point&p, const vecteur &v)
{
	point p2 = {p.x + v.x, p.y + v.y, p.z + v.z};
	return p2;
}

inline point operator - (const point&p, const vecteur &v)
{
	point p2 = {p.x - v.x, p.y - v.y, p.z - v.z};
	return p2;
}

inline vecteur operator  ^ (const vecteur &u,const vecteur &v)
{	   
	vecteur ret;

	ret.x = ( u.y * v.z ) - ( u.z * v.y );
	ret.y = ( u.z * v.x ) - ( u.x * v.z );
	ret.z = ( u.x * v.y ) - ( u.y * v.x ); 

	return ret;
}


inline vecteur operator + (const vecteur&v1, const vecteur &v2)
{
	vecteur v = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
	return v;
}

inline vecteur operator - (const point&p1, const point &p2)
{
	vecteur v = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
	return v;
}

inline vecteur operator * (float c, const vecteur &v)
{
	vecteur ret = {v.x *c, v.y * c, v.z * c};
	return ret;
}

inline vecteur operator - (const vecteur&v1, const vecteur &v2)
{	
	vecteur v = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
	return v;
}

inline float operator * (const vecteur&v1, const vecteur &v2 ) 
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline color operator * (const color&c1, const color &c2 )
{
	color c = {c1.red * c2.red, c1.green * c2.green, c1.blue * c2.blue};
	return c;
}

inline color operator + (const color&c1, const color &c2 ) 
{
	color c = {c1.red + c2.red, c1.green + c2.green, c1.blue + c2.blue};
	return c;
}

inline color operator * (float coef, const color &c )
{
	color c2 = {c.red * coef, c.green * coef, c.blue * coef};
	return c2;
}

#endif //__DEF_H
