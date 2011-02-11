#include "scene.h"
#include "display.h"



int main (int argc, char *argv[])
{
	if(!CScene::getSingletonPtr()->initScene())
	{
		return -1;
	}
	
	if(!initGL(argc,argv))
	{
		return -2;
	}
	
	if (!initGLBuffers())
	{
		return -3;
	}

	//glutSetCursor(GLUT_CURSOR_NONE);

	glutKeyboardFunc(eventClavier);

	glutMotionFunc(eventSourisMouv);
	glutMouseFunc(eventSouris);


	/*glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_RGB);

	glutInitWindowPosition(200,200);
	
	glutInitWindowSize(CScene::getSingletonPtr()->sizex,CScene::getSingletonPtr()->sizey);

	glutCreateWindow("Raytracer");
	
	glutDisplayFunc(DrawFunc);

	//glutReshapeWindow(CameraSettings::getSingletonPtr()->sceneCfg.sizex,CameraSettings::getSingletonPtr()->sceneCfg.sizey);

	glutWarpPointer(CScene::getSingletonPtr()->sizex/2,CScene::getSingletonPtr()->sizey/2);


*/
	//glutTimerFunc(5,MAJ,0);
	
	glutMainLoop();

    return 0;
}