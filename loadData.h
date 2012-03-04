/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: loadData.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * LoadRawFile
 *
 * Description: 
 * Loading the raw volume and returning a pointer to the containing 
 * array. 
 *
 * Formal Parameters:
 * File name (char*) : Name of the loaded file, 
 * File size (size_t) : Size of the loaded file. 
 *
 * Returns:
 * Volume Data (char*) : Pointer to the loaded volume. 
 * 
 * Note(s):
 * fVolumeData is internal. 
 ********************************************************************/ 
char* LoadRawFile(char* fFileName, size_t fSize)
{
	/* Pointer to the file */
	FILE* fPtrFile; 
	
	/* Open the file */
	fPtrFile = fopen(fFileName, "rb"); 
	
	/* Check for a valid file */
	if (!fPtrFile)
	{
		fprintf(stderr, "Error Opening File '%s' ... \n", fFileName); 
		exit(0); 
	}
	
	/* Volume array */   
	char* fVolumeData = (char*) malloc (fSize); 
	
	/* Read the volume file into the array */
	size_t fFileSizeBytes = fread(fVolumeData, 1, fSize, fPtrFile); 
	
	/* Volume array */
	return fVolumeData; 
}

/********************************************************************
 * Name: 
 * InitData
 *
 * Description: 
 * Heigher level function for loading the volume.   
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void. 
 *
 * Note(s): 
 * mVolumeData is global. 
 * mPath is global. 
 ********************************************************************/
void InitData()
{
	printf("Loading DataSet ... \n"); 
	
	/* Path checking */
	if (mPath == 0) 
	{
		printf("Error Finding File '%s' ... \n", mPath); 
		exit(0); 
	}
	 
	/* Loading Data from the file specified by the mPath */  
	mVolumeData = LoadRawFile(mPath, mVolumeSizeBytes_DS); 
	
	printf("	DataSet Loaded Successfully \n\n");
}


/********************************************************************
 * Name: 
 * AllocateVolumeArrays
 *
 * Description: 
 * 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 * volIn3D, volOut3D, mExtractedVolumeData are globals.
 ********************************************************************/
void AllocateVolumeArrays()
{ 
	/* Allocating input volume 3D array */
	volIn3D = (char***) malloc(mVolWidth_DS * sizeof(char**));
	for (int y = 0; y < mVolWidth_DS; y++)
	{
		volIn3D[y] = (char**) malloc (mVolHeight_DS * sizeof(char*));
		for (int x = 0; x < mVolHeight_DS; x++)
		{
			volIn3D[y][x] = (char*) malloc(mVolDepth_DS * sizeof(char));
		}
	}
	
	/* Allocating output volume 3D array */
	volOut3D = (char***) malloc(mVolWidth * sizeof(char**));
	for (int y = 0; y < mVolWidth; y++)
	{
		volOut3D[y] = (char**) malloc (mVolHeight * sizeof(char*));
		for (int x = 0; x < mVolHeight; x++)
		{
			volOut3D[y][x] = (char*) malloc(mVolDepth * sizeof(char));
		}
	}
	
	/* Allocating extracted volume array */
	mExtractedVolumeData = (char*) malloc ( mVolumeSize * sizeof(char)); 
}

/********************************************************************
 * Name: 
 * ExtractVolume
 *
 * Description: 
 * Allocating 3D arrays for the loaded input volume, the extracted 
 * subvolume, and 1D array for the extracted subvolume. 
 * Extracting the subvolume from the loaded volume.
 * Repacking the subvolume in 1D array. 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 * mVolumeData, volIn3D, volOut3D, mExtractedVolumeData are globals.
 ********************************************************************/
void ExtractVolume()
{	
	/* Allocating input volume, output volume, extracted volume arrays */ 
	AllocateVolumeArrays(); 

	/* Moving the 3D volume from the input 1D array mVolumeData to the 
	 * 3D array volIn3D */ 
	int ctr = 0; 
	for (int i = 0; i < mVolWidth_DS; i++)
	{
		for (int j = 0; j < mVolHeight_DS; j++)
		{
			for (int k = 0; k < mVolDepth_DS; k++)
			{
				volIn3D[i][j][k] = mVolumeData[ctr]; 
				ctr++; 
			}
		}
	}

	/* Starting point (X, Y, Z) to extract the smaller subvolume */
	int startX = 0;
	int startY = 0;
	int startZ = 0;

	/* Extract smaller subvolume from the original loaded dataset */ 
	for (int i = 0; i < mVolWidth; i++)
	{
		for (int j = 0; j < mVolHeight; j++)
		{
			for (int k = 0; k < mVolDepth; k++)
			{
				volOut3D[i][j][k] = volIn3D[(startX) + i][(startY) + j][(startZ) + k];
				
			}
		}
	}

	/* Repack the extracted Volume in 1D array for later usage */ 
	ctr = 0;
	for (int i = 0; i < mVolWidth; i++)
	{
		for (int j = 0; j < mVolHeight; j++)
		{
			for (int k = 0; k < mVolDepth; k++)
			{
				mExtractedVolumeData[ctr] = volOut3D[i][j][k]; 
				ctr++; 
			}
		}
	}
} 

/********************************************************************
 * Name: 
 * CreateFloatData
 *
 * Description: 
 * Allocating 3D arrays for the loaded input volume, the extracted 
 * subvolume, and 1D array for the extracted subvolume. 
 * Extracting the subvolume from the loaded volume.
 * Repacking the subvolume in 1D array. 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 ********************************************************************/
void CreateFloatData()
{
	/* Allocate float array for the input volume to increase 
	 * precision */
	mVolumeDataFloat = (float*) malloc ( mVolumeSize * sizeof(float)); 
	
	/* Extract the subvolume from the loaded volume */
	ExtractVolume(); 
	
	/* Packing the extracted volume in the float 1D array */ 
	printf("Packing Data in a Float Array ... \n"); 
	for (int i = 0; i < mVolumeSize; i++)
		mVolumeDataFloat[i] = (float) (unsigned char) mExtractedVolumeData[i];
	printf("	Packing is Done  \n");
	
	/* Realeasing volume in char array */  
	printf("	Releasing Byte Data 	\n\n");
	delete [] mVolumeData; 
}


/********************************************************************
 * Name: 
 * CreateSpectrum
 *
 * Description: 
 * 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 * mVolumeData, volIn3D, volOut3D, mExtractedVolumeData are globals.
 ********************************************************************/
void CreateSpectrum()
{
	printf("Creating Complex Spectrum ... \n"); 	
	
	/* Allocating single precision complex array for the created 
	 * spectrum */  
	mVolumeArrayComplex = (fftwf_complex*) fftwf_malloc (mVolWidth * mVolHeight * mVolDepth * sizeof(fftwf_complex)); 
	
	/* Packing the final volume in the complex array 
	 * Real component only, the imaginary one is left untill getting 
	 * the spectrum */  
	printf("	Packing Volume Array for Single Precision Format ... \n"); 
	for (int i = 0; i < mVolumeSize; i++)
	{
		mVolumeArrayComplex[i][0] = mVolumeDataFloat[i]; 
		mVolumeArrayComplex[i][1] = 0;
	}
	printf("	Packing Done Successfully \n\n"); 
	
	/* Fourier transforming the spatial volume and getting the spectral 
	 * one */ 
	printf("	Executing 3D Forward FFT  ... \n"); 
	mFFTWPlan = fftwf_plan_dft_3d(mVolWidth, mVolHeight, mVolDepth, mVolumeArrayComplex, mVolumeArrayComplex, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(mFFTWPlan);
	printf("	3D Forward FFT Done Successfully  \n\n");	
}

/********************************************************************
 * Name: 
 * PackingSpectrumTexture
 *
 * Description: 
 * 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 * mVolumeData, volIn3D, volOut3D, mExtractedVolumeData are globals.
 ********************************************************************/
void PackingSpectrumTexture()
{
	/* Allocating a texture array to suit a 3D OpenGL 32 bit texture 
	 * with 2 components RG */
	mTextureArray = (float*) malloc (mVolumeSize * 2 * sizeof(float)); 
	
	printf("Packing Spectrum into Texture Array ... \n");
	int ctr = 0; 
	for (int i = 0; i < (mVolumeSize * 2); i += 2)
	{	
		mTextureArray[i]		= mVolumeArrayComplex[ctr][0];
		mTextureArray[i + 1]	= mVolumeArrayComplex[ctr][1];
		ctr++; 
	}
	
	printf("	Packing Spectrum Into Texture Done Successfully \n\n");
}

/********************************************************************
 * Name: 
 * SendSpectrumTextureToGPU
 *
 * Description: 
 * 
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.
 *
 * Note(s):
 * mVolumeData, volIn3D, volOut3D, mExtractedVolumeData are globals.
 ********************************************************************/
void SendSpectrumTextureToGPU(void)
{
	printf("Creating & Binding Spectrum Texture To GPU ... \n"); 

	/* 3D OpenGL texture creation & binding */  
	glGenTextures(1, &mVolTexureID);
	glBindTexture(GL_TEXTURE_3D, mVolTexureID);
	
	/* Texture parameters */
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	
	/* For Automatic Texture Coordinate Generation */
	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
	 
	/* Upload the spectrum array to GPU texture memory */  
	printf("	Transfer Spectrum to GPU Memory ... \n");
	glTexImage3D(GL_TEXTURE_3D, 0, RG32F, mVolWidth, mVolHeight, mVolDepth, 0, RG, GL_FLOAT, mTextureArray );
	printf("	Transfering Data to GPU Memory Done Successfully \n\n"); 

	
	/*
	 cudaError_t err = cudaGraphicsGLRegisterImage
	(&volTextureBufferResource, mVolTexureID, GL_TEXTURE_3D, cudaGraphicsMapFlagsNone);

	 printf("cudaGraphicsGLRegisterImage error [%d]:",err);
	 if ( err==cudaSuccess) printf( "cudaSuccess" ); 
	 if ( err==cudaErrorInvalidDevice) printf( "cudaErrorInvalidDevice" ); 
	 if ( err==cudaErrorInvalidValue) printf( "cudaErrorInvalidValue" ); 
	 if ( err==cudaErrorInvalidResourceHandle) printf( "cudaErrorInvalidResourceHandle" ); 
	 if ( err==cudaErrorUnknown) printf( "cudaErrorUnknown" ); 
	 printf("\n");
	 */
}

