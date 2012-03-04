/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fourierVolumeRenderer.cpp
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"
#include "contextCUDA.h"
#include "contextOpenGL.h"
#include "buffersCreateDelete.h"
#include "contextClean.h"
#include "initVariables.h"
#include "loadData.h"
#include "wrappingAround.h"
#include "fftShift.h"
#include "displayList.h"
#include "cuda3D.h"

/* If GPU, run the pure GPU-based renderer, else run the hybrid one. 
 * Toggle with SPACE */
bool GPU = true;

/********************************************************************
 * Name: 
 * main
 *
 * Description: 
 * Project entry. 
 *
 * Formal Parameters:
 * argc (int) : Name of input arguments, 
 * argv (char*) : Argumetns. 
 *
 * Returns:
 * 0 (int) : main return.  
 * 
 * Note(s):
 ********************************************************************/
int main(int argc, char** argv)
{	
	printf("--------------------------------------------- \n");
	printf("Starting Fourier Volume Rendering On CUDA ... \n");
	printf("--------------------------------------------- \n\n");
	
	/* Initializing necessary variables */ 
	InitVars(); 
	
	/* Initializing OpenGL & CUDA contexts */ 
	/* First initialize OpenGL context, so we can properly set the 
	 * GL for CUDA.
	 * This is necessary in order to achieve optimal performance with 
	 * OpenGL/CUDA interop. */ 
	if( false == InitOpenGLContext(&argc, argv))
	{
		printf("OpenGL Context Not Created Successfully \n");
		exit (0);
	}
	else 
	{
		printf("    OpenGL Context Created Successfully - Moving to CUDA Context \n\n"); 
		printf("------------------------------------------------------------------------------ \n\n");
	}
	
	/* Initialize CUDA context (GL context has been created already) */ 
	InitCUDAContext(argc, argv, true);
	
	/* Reading & initializing volume data */ 	
	InitData(); 
	
	/* Creating float volume & releasing byte data */ 	
	CreateFloatData();

	/* CUDA or Hybrid pipeline */
	if (!GPU)
	{
		/* Wrapping around spatial volume */  
		WrapAroundVolume(); 

		/* Creating spectrum complex arrays */ 
		CreateSpectrum(); 
		
		/* Wrapping around spectral volume */  
		WrapAroundSpectrum(); 
		
		/* Packing spectrum complex data into texture array to 
		 * be sent to OpenGL */ 
		PackingSpectrumTexture(); 	
	}
	else 
	{
		/* Spectral texture for OpenGL compatability */
		mTextureArray = (float*) malloc (mVolumeSize * 2 * sizeof(float)); 
		
		/* Run the FVR on the CUDA pipeine */
		CUDA_Way();
	}

	/* Uploading spectrum texture to GPU for slicing */
	SendSpectrumTextureToGPU(); 
	
	/* We don't need float data ayn more as it resides in the 
	 * GPU texture memory */ 	
	delete [] mVolumeDataFloat; 
	
	/* Intersecting QUAD with the texture */  
	SetDisplayList();
	
	/* CUDA timer */ 
	cutCreateTimer(&mTimer);
	cutResetTimer(mTimer);

	/* Register OpenGL callbacks */ 
	glutDisplayFunc(DisplayGL);
	glutKeyboardFunc(KeyBoardGL);
	glutReshapeFunc(ReshapeGL);
	glutIdleFunc(IdleGL);
	
	/* Initializing OpenGL buffers */ 
	InitOpenGLBuffers();
	
	/* Start main rendering loop */ 
	glutMainLoopß();

	/* Clean Up */ 
	CleanUp(EXIT_FAILURE);

	/* Exiting ... */ 
	// shrEXIT(argc, (const char**)argv);
}