/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering 
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fourierVolumeRenderer.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: Main header file for the project. 
 * Note(s)		: 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"

// Creating Extracted Projection-Slice Input Texture to CUDA 
void CreateTextureSrc(GLuint* fTextureID, unsigned int sizeX, unsigned int sizeY)
{
	// Generating Texture ID 
	glGenTextures(1, fTextureID);
	
	// Binding Texture 
	glBindTexture(GL_TEXTURE_2D, *fTextureID);
	
	// Texture Parameters 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
	// Allocate 2D 2-Component Texture on the GPU 
	glTexImage2D(GL_TEXTURE_2D, 0, RG32F, sizeX, sizeY, 0, RG, GL_FLOAT, NULL);

	// Check for CUDA / OpenGL Errors
	// CUT_CHECK_ERROR_GL2(); 

	// Register Texture with CUDA
	cudaGraphicsGLRegisterImage
	(&CUDASliceResourceBuffer, *fTextureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);              
}  

// Create Destination Texture to Render the Image to
void CreateTextureDst(GLuint* fTextureID, unsigned int sizeX, unsigned int sizeY)
{
	// Getting Texture ID 
	glGenTextures(1, fTextureID);
	
	// Binding Texture 
	glBindTexture(GL_TEXTURE_2D, *fTextureID);
	
	// Texture Parameters 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Allocating Texture on the GPU (OpenGL Side)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, sizeX, sizeY, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    	
	// Check OpenGL/CUDA Errors
	// CUT_CHECK_ERROR_GL2();

	// Registering Texture with CUDA
	cudaGraphicsGLRegisterImage
	(&CUDA_ResultingTextureBufferResource, *fTextureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
}

// Create Destination Texture for the 3D Spectral Volume 
void CreateVolumeTextureDst(GLuint* fTextureID, unsigned int sizeX, unsigned int sizeY)
{	
	// Getting Texture ID 
	glGenTextures(1, fTextureID);
	
	// Binding Texture 
	glBindTexture(GL_TEXTURE_3D, *fTextureID);
	
	// Texture Parameters (Wrapping)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	
	// Texture Parameters (Interpolation)
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	
	// For Automatic Texture Coordinate Generation
	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);

	// Allocating Texture on the GPU (OpenGL Side)
	glTexImage3D(GL_TEXTURE_3D, 0, RG32F, mVolWidth, mVolHeight, mVolDepth, 0, RG, GL_FLOAT, NULL);
    	
	// Check OpenGL/CUDA Errors
	// CUT_CHECK_ERROR_GL2();

	// Registering Texture with CUDA
	//cudaGraphicsGLRegisterImage
	//(&CUDA_VolumeTextureBufferResource, *fTextureID, GL_TEXTURE_3D, cudaGraphicsMapFlagsWriteDiscard);
}

// Creating cuFFT Buffers on CUDA 
void CreateBuffersFFT()
{
	cutilSafeCall(cudaMalloc((void**)&cuFFT_Input, sizeof(cufftComplex) * mSliceWidth * mSliceHeight));
	cutilSafeCall(cudaMalloc((void**)&cuFFT_Output, sizeof(cufftComplex) * mSliceWidth * mSliceHeight));
	cutilSafeCall(cudaMalloc((void**)&TempArray, sizeof(float) * mSliceWidth * mSliceHeight));
}

// Deleting Texture
void DeleteTexture(GLuint* fTextureID)
{
	glDeleteTextures(1, fTextureID);
	// CUT_CHECK_ERROR_GL2();
	*fTextureID = 0;
}


void CreateFrameBuffer(GLuint* fFBO_ID, GLuint fColor)
{
	// Get FBO ID
	glGenFramebuffersEXT(1, fFBO_ID);
	
	// Bind FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fFBO_ID);

	// Attach Images
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fColor, 0);
	
	// Unbind that FrameBuffer  
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	
	// Check OpenGL/CUDA Errors 
	// CUT_CHECK_ERROR_GL2();
}

void DeleteFrameBuffer(GLuint* fFBO_ID)
{
	glDeleteFramebuffersEXT(1, fFBO_ID);
	// CUT_CHECK_ERROR_GL2();
	*fFBO_ID = 0;
}

// OpenGL Buffers Initialization 
void InitOpenGLBuffers()
{
	// Create PBO
	CreatePBO(&PBODestinationBuffer, &CUDA_PBODestinationBufferResource); 
	
	// Create Texture that will Receive the Result of CUDA
	CreateTextureDst(&ImgTextureID, mImageWidth, mImageHeight);

	// Create Texture for Blitting onto the Screen
	CreateTextureSrc(&SliceTextureID, mImageWidth, mImageHeight);
	
	// Create CUDA FFT Arrays on the GPU
	CreateBuffersFFT(); 

	// Create a FrameBuffer for Off-Screen Rendering
	CreateFrameBuffer(&FrameBufferID, SliceTextureID);

	// Check OpenGL/CUDA Errors 
    // CUT_CHECK_ERROR_GL2();
}

// Creating PBO
void CreatePBO(GLuint* fPBO, struct cudaGraphicsResource **fPBO_Resource)
{
	// Setup Fragment Data Parameters
	mNumTexels = mWindowWidth * mWindowHeight;
	mNumValues = mNumTexels * 1;
	mSizeTextureData = sizeof(float) * mNumValues;
	void *data = malloc(mSizeTextureData);

	// Create Buffer Object
	glGenBuffers(1, fPBO);
	glBindBuffer(GL_ARRAY_BUFFER, *fPBO);
	glBufferData(GL_ARRAY_BUFFER, mSizeTextureData, data, GL_DYNAMIC_DRAW);
	free(data);

	// Release Buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register this Buffer Object with CUDA
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(fPBO_Resource, *fPBO, cudaGraphicsMapFlagsNone));
	// CUT_CHECK_ERROR_GL2();
}

// Deleting PBO
void DeletePBO(GLuint* fPBO)
{
	// Delete the Buffer 
	glDeleteBuffers(1, fPBO);
	
	// Check OpenGL/CUDA Errors 
	// CUT_CHECK_ERROR_GL2();
	
	// No Dangling Pointers 
	*fPBO = 0;
}

