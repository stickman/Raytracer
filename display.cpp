#include "display.h"
#include "Scene.h"
#include "Camera.h"
#include "Quaternion.h"

// Particle data
GLuint vbo = 0;

static int fpsCount = 0;
static int fpsLimit = 1;
unsigned int timer;
float ifps;

static float *pixels = NULL;
int x_old;
int y_old;
bool click = false;

void eventSouris(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		if(!click)
		{
			click = true;
			x_old = x;
			y_old = y;
		}

	}
	else if(state == GLUT_UP )
	{
		click = false;
		x_old = x;
		y_old = y;
	}
}

void eventSourisMouv(int x, int y)
{
	if (click)
	{
		CScene * pscene = CScene::getSingletonPtr();

		int deltaX = (x-x_old )*2;
		int deltaY = (y-y_old )*2;
		
		if (deltaX!=0)
		{
			YAW(pscene->mCamera.vXAxis,pscene->mCamera.vYAxis,pscene->mCamera.vZAxis,0.03 * deltaX/ifps);
		x_old = x;
		}

		if (deltaY!=0)
		{
			roll(pscene->mCamera.vXAxis,pscene->mCamera.vYAxis,pscene->mCamera.vZAxis, 0.03 * deltaY/ifps);
			y_old = y;
		}
	}
}

void eventClavier(unsigned char key ,int x,int y)
{
	
	float mouvement = 200/ifps;

	CScene * pscene = CScene::getSingletonPtr();

	switch(key)
	{
	case(27):
		exit(0);
		break;
	case ('z'):		
		pscene->mCamera.position+=mouvement*pscene->mCamera.vZAxis;
		break;
	case ('s'):		
		pscene->mCamera.position-=mouvement*pscene->mCamera.vZAxis;
		break;
	case ('q'):
		pscene->mCamera.position-=mouvement*pscene->mCamera.vXAxis;
		break;
	case ('d'):
		pscene->mCamera.position+=mouvement*pscene->mCamera.vXAxis;
		break;
	}
	
}

void process() 
{
    float *out_data;
	CScene* pScene=CScene::getSingletonPtr();

    // map buffer objects to get CUDA device pointers
    cutilSafeCall(cudaGLMapBufferObject( (void**)&out_data, vbo));
    cutilCheckMsg("cudaGLMapBufferObject failed");

    // calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(pScene->sizex / block.x, pScene->sizey / block.y, 1);


    // execute CUDA kernel
	launch_cudaProcess(grid, block, out_data, pScene->sizex,pScene->sizey,
		pScene->mCamera.position,pScene->mCamera.vXAxis,pScene->mCamera.vYAxis,pScene->mCamera.vZAxis,
		pScene->materialContainer,
		pScene->sphereContainer,
		pScene->lightContainer,
		pScene->mCubeMap,
		pScene->pRootBSP,pScene->pMemCuda);
    
	cutilCheckMsg("launch_cudaProcess\n.");

    cutilSafeCall(cudaGLUnmapBufferObject( vbo));
    cutilCheckMsg("cudaGLUnmapBufferObject failed");
}


bool initGL(int argc, char **argv)
{
	CScene* pScene=CScene::getSingletonPtr();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(pScene->sizex, pScene->sizey);
    glutCreateWindow("RayTracerForCuda");
    glutDisplayFunc(display);

    glewInit();
    if (! glewIsSupported(
        "GL_ARB_vertex_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return false;
    }
	
    cutCreateTimer(&timer);
    cutResetTimer(timer); 

    return true;
}

void display(void)
{     
	CScene* pScene=CScene::getSingletonPtr();

	cutStartTimer(timer);

	process();

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, vbo);
	
	glDrawPixels(pScene->sizex,pScene->sizey,GL_RGBA,GL_FLOAT,pixels);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	cutStopTimer(timer);

    glutSwapBuffers();

	fpsCount++;
	if (fpsCount == fpsLimit) {
       char fps[256];
       ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
       sprintf(fps, "CUDA GL Post Processing(%d x %d): %3.1f fps", pScene->sizex, pScene->sizey, ifps);  
       //printf("CUDA GL Post Processing(%d x %d): %3.1f fps\n",pScene->sizex, pScene->sizey, ifps);
       glutSetWindowTitle(fps);
       fpsCount = 0; 
	}
	glutPostRedisplay();
	
}

bool initGLBuffers()
{
	CScene* pScene=CScene::getSingletonPtr();

	glGenBuffersARB(1, &vbo);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);

	glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(float) * pScene->sizex * pScene->sizey * 4, pixels , GL_DYNAMIC_DRAW_ARB);

	int bsize=0;

    glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bsize); 


	if (bsize != sizeof(float) * pScene->sizex * pScene->sizey * 4)
	{
		return false;
	}

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    cudaGLRegisterBufferObject(vbo);

    cutilCheckMsg("cudaGLRegisterBufferObject failed");

	return true;

}
