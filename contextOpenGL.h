/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: contextOpenGL.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/  

#include "fourierVolumeRenderer.h"

/* Debugging */
// #define _DEBUG_ALL_ 


float mImageScale = 2; 

/********************************************************************
 * Name: 
 * InitOpenGLContext
 *
 * Description: 
 * OpenGL context initialization 
 * 
 * Formal Parameters:
 * argc (int) : Number of inout arguments, 
 * argv size (char*) : Arguments. 
 *
 * Returns:
 * (bool) : True if initialization was valid. 
 * 
 * Note(s):
 ********************************************************************/ 
bool InitOpenGLContext(int *argc, char **argv )
{
	printf("--------------------------- \n");
	printf("Creating OpenGL Context ... \n");
	printf("--------------------------- \n");

	if (IsOpenGLAvailable(sSDKname)) 
		printf("    OpenGL Device is Available & Ready for Usage \n");
	else 
	{
		printf("    OpenGL Device is NOT Available, Exiting \n\n");
		return false;
	}

	/* GLUT initialization */
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(mWindowWidth, mWindowHeight);

	/* GLUT window creation */  
	iGLUTWindowHandle = glutCreateWindow(" Fourier Volume Rendering on CUDA");

	printf("    GLUT Initialized Successfully \n");
	printf("    GLUT Double Buffering \n");
	printf("    GLUT Window Dimensions = [%d] x [%d] Pixels \n\n", mWindowWidth, mWindowHeight);
	
	/* Initialize necessary OpenGL extensions via GLEW library */ 
	glewInit();

	printf("    GLEW Initialized Successfully \n");
	
	/* Required OpenGL extensions*/
	if (! glewIsSupported("GL_VERSION_2_0 " 
						  "GL_ARB_pixel_buffer_object " 
						  "GL_EXT_framebuffer_object "))
	{
		printf("    ERROR: Support for Necessary OpenGL Extensions Missing \n\n");
		fflush(stderr);
		return CUTFalse;
	}
	
	return CUTTrue;
}

/********************************************************************
 * Name: 
 * ExtractSlice
 *
 * Description: 
 * Extract projection-slice from OpenGL 3D spectral texture. This 
 * function runs in the off-screen OpenGL context.
 * 
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
void ExtractSlice()
{
#ifdef _DEBUG_ALL_
	printf("Extracting Central Slice from the 3D Spectrum \n");
#endif

	/* Clear buffer */ 
	glClear(GL_COLOR_BUFFER_BIT);

	/* Enable 3D texturing */ 
	glEnable(GL_TEXTURE_3D);

	/* Replacing the old projection-slice */ 
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	
	/* Bind the 3D spectral texture */
	glBindTexture(GL_TEXTURE_3D, mVolTexureID);
	
	/* Adjust OpenGL view-port - Marawanism :) */
	glViewport(-(mUniDim / 2),-(mUniDim / 2),(mUniDim * 2),(mUniDim * 2));
	
	/* Texture corrdinate generation */ 
	glEnable(GL_TEXTURE_GEN_S);
	glEnable(GL_TEXTURE_GEN_T);
	glEnable(GL_TEXTURE_GEN_R);
	
	/* Loading identity to projection matrix */ 
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	/* Loading identity to model-view matrix */
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* Define the main six clip planes */
	static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
	static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
	static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
	static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
	static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
	static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};

	/* Automatic texture coordinate generation */
	static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
	static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
	static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

	/* Saving scene state */ 
	glPushMatrix ();
	
	 
	/* Transform (Rotation Only) the Viewing Direction
	 * We don't need except the - 0.5 translation in each dimension to adjust 
	 * the Texture in the center of the scene i.e. (3D k-space)
	 */ 
	glRotatef(-mXrot, 0.0, 0.0, 1.0);
	glRotatef(-mYrot, 0.0, 1.0, 0.0);
	glRotatef(-mZrot, 1.0, 0.0, 0.0);
	glTranslatef(-0.5, -0.5, -0.5);
	
	/* Automatic texture coordinates generation */
	glTexGenfv(GL_S, GL_EYE_PLANE, x);
	glTexGenfv(GL_T, GL_EYE_PLANE, y);
	glTexGenfv(GL_R, GL_EYE_PLANE, z);

	/* Define the main six clip planes */
	glClipPlane(GL_CLIP_PLANE0, eqx0);
	glClipPlane(GL_CLIP_PLANE1, eqx1);
	glClipPlane(GL_CLIP_PLANE2, eqy0);
	glClipPlane(GL_CLIP_PLANE3, eqy1);
	glClipPlane(GL_CLIP_PLANE4, eqz0);
	glClipPlane(GL_CLIP_PLANE5, eqz1);

	/* Restore the State */ 
	glPopMatrix ();

	/* Enable Clip Planes */
	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE1);
	glEnable(GL_CLIP_PLANE2);
	glEnable(GL_CLIP_PLANE3);
	glEnable(GL_CLIP_PLANE4);
	glEnable(GL_CLIP_PLANE5);

	/* Render enclosing rectangle at (0,0) that represents the 
	 * extracted projection-slice */
	glCallList(mDiaplayList);
	glPopMatrix();  

	/* Disable texturing */  
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);
	
	/* Unloading clip-planes */ 
	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);
	glDisable(GL_CLIP_PLANE3);
	glDisable(GL_CLIP_PLANE4);
	glDisable(GL_CLIP_PLANE5);
	
	/* Unbind 3D spectral texture texture */  
	glBindTexture(GL_TEXTURE_3D, 0);

#ifdef _DEBUG_ALL_
	printf("    Central Slice Extracted Successfully \n"); 
#endif
}

/********************************************************************
 * Name: 
 * DisplayGLImage
 *
 * Description: 
 * Display reconstructed image as Textured Quad.
 * 
 * Formal Parameters:
 * fTexture (GLuint) : Texture ID. 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
// 
void DisplayGLImage(GLuint fTexture)
{
	/* Clearing color buffer */ 	
	glClear(GL_COLOR_BUFFER_BIT);

	/* Bind Texture to Target */
	glBindTexture(GL_TEXTURE_2D, fTexture);
	
	/* Enable texturing */ 
	glEnable(GL_TEXTURE_2D);
	
	/* Diable depth buffering & lighting */ 
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	
	/* Replace the old image with the new one */ 
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	/* Adjust OpenGL viewport again */
	glViewport(0, 0, mWindowWidth, mWindowWidth); 

	/* Loading identity to projection matrix */ 
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	/* Loading identity to model-view matrix */
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	/* Saving state */
	glPushMatrix ();

	/* Scale the reconstructed image to fit on the GLUT window */
	glScalef(mImageScale, mImageScale, 1); 
	glTranslatef(-0.5, -0.5, 0.0);
	
	/* Plot or map the reconstructed image on the QUAD */ 
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0); glVertex3f(0, 0, 0);
		glTexCoord2f(1.0, 0.0); glVertex3f(1.0, 0, 0);
		glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0);
		glTexCoord2f(0.0, 1.0); glVertex3f(0.0, 1.0, 0);
	glEnd();
	
	/* Restore the scene state */ 
	glPopMatrix();
	
	/* Disable 2D texturing */ 
	glDisable(GL_TEXTURE_2D);

	/* Unbind texture */
	glBindTexture(GL_TEXTURE_2D, 0);
}

/********************************************************************
 * Name: 
 * DisplayGL
 *
 * Description: 
 * OpenGL display callback. 
 * 
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
void DisplayGL()
{
	/* Starting CUDA timer */ 
	cutStartTimer(mTimer);
	
	/* Enable CUDA prjection-slice processing or not */
    if (EnableCUDA) 
    {
        /* Bind frame buffer and render to attached texture */
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, FrameBufferID);
        
		/* Extract projection-slice from 3D spectrum */  
		ExtractSlice();

		/* Interpolate the projection-slice */
        ProcessSlice();
		
		/* Unbind the frame buffer */ 
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
        
		/* Diaply the reconstructed image */  
		DisplayGLImage(ImgTextureID);
    }
    else
	{
        /* Just make the slice extraction & draw the normal spectrum */  
		ExtractSlice();
	}
    
	/* CUDA synchronization */ 
	cudaThreadSynchronize();
	
	/* Stop timer & profile */
	cutStopTimer(mTimer);

    /* Flip back-buffer */
    glutSwapBuffers();
    
    /* Update fps counter, fps/title DisplayGL */
    if (++mFPSCount == mFPSLimit)
	{
        char cTitle[256];
        float fps = 1000.0f / cutGetAverageTimerValue(mTimer);
        sprintf(cTitle, "CUDA Fourier Volume Rendering (%d x %d): %.1f fps", mWindowWidth, mWindowHeight, fps);  
        glutSetWindowTitle(cTitle);
        
        mFPSCount = 0; 
        mFPSLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        cutResetTimer(mTimer);  
    }
}

/********************************************************************
 * Name: 
 * IdleGL
 *
 * Description: 
 * OpenGL idle function. 
 * 
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
void IdleGL()
{
    glutPostRedisplay();
}

/********************************************************************
 * Name: 
 * KeyBoardGL
 *
 * Description: 
 * OpenGL keybord function. 
 * 
 * Formal Parameters:
 * key (unsigned char) : Input key, (int), (int). 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
void KeyBoardGL(unsigned char key, int, int)
{
	switch(key) 
	{
		/* Escape */
		case(27) : CleanUp(EXIT_SUCCESS);
			break; 
		
		/* Switching between GPU-based render and the hybrid one */
		case ' ' : EnableCUDA ^= 1;
			break;
		
		/* X-axis rotation increment */
		case 'q' : mXrot += 5.0; 
			break; 
		
		/* X-axis rotation decrement */
		case 'Q' : mXrot -= 5.0; 
			break;
		
		/* Y-axis rotation increment */
		case 'w' : mYrot += 5.0; 
			break; 
		
		/* Y-axis rotation decrement */
		case 'W' : mYrot -= 5.0; 
			break;
		
		/* Z-axis rotation increment */
		case 'e' : mZrot += 5.0; 
			break; 
		
		/* Z-axis rotation decrement */
		case 'E' : mZrot -= 5.0; 
			break;

		/* Intensity scale factor scaling up (* 5)*/
		case 'a' : mScaleFactor *= 5.0; 
			break;

		/* Intensity scale factor scaling down (/ 5)*/
		case 'A' : mScaleFactor /= 5.0; 
			break;
			
		/* Intensity scale factor increment (+ 10) */
		case 's' : mScaleFactor += 10.0; 
			break;
			
		/* Intensity scale factor decrement (- 10) */
		case 'S' : mScaleFactor -= 10.0; 
			break;
		
		/* Image size scale up */
		case 'z' : mImageScale += 0.5; 
			break;
		
		/* Image size scale down */
		case 'Z' : mImageScale -= 0.5; 
			break;
		
	}
	
	/* GLUT re-draw */
	glutPostRedisplay(); 
}

/********************************************************************
 * Name: 
 * ReshapeGL
 *
 * Description: 
 * OpenGL reshape function. 
 * 
 * Formal Parameters:
 * fWinWidth (int) : GLUT window width, 
 * fWinHeight (int) : GLUT window height. 
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/ 
void ReshapeGL(int fWinWidth, int fWinHeight)
{
    mWindowWidth 	= fWinWidth;
    mWindowHeight 	= fWinHeight;
}
