/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: contextClean.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/  

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * CleanUp
 *
 * Description: 
 * Cleanning context stuff and exit contexts safely.    
 *
 * Formal Parameters:
 * iExitCode (int) : Exit code. 
 *
 * Returns:
 * void. 
 * 
 * Note(s):
 ********************************************************************/
void CleanUp(int iExitCode)
{
	/* Releasing CUDA timer */  
	cutilCheckError(cutDeleteTimer(mTimer));

	/* Unregister buffer object with CUDA */ 
	cutilSafeCall(cudaGraphicsUnregisterResource(CUDASliceResourceBuffer));
	cutilSafeCall(cudaGraphicsUnregisterResource(CUDA_PBODestinationBufferResource));
	
	/* Delete all OpenGL buffers */ 
	DeletePBO(&PBODestinationBuffer);
	DeleteTexture(&SliceTextureID);
	DeleteTexture(&ImgTextureID);
	DeleteFrameBuffer(&FrameBufferID);
	
	/* Exitting CUDA context safely */ 
	cudaThreadExit();
	
	/* Destroy GLUT window if still exisiting */ 
	if(iGLUTWindowHandle)
		glutDestroyWindow(iGLUTWindowHandle);

	/* Exiting */
	exit (iExitCode);
}

