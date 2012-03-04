/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: cuda3D.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"

/* Controling the size of the final volume to be processed */
int sDataSize = 256; 

/** CUDA FFT arrays **/
cufftComplex* sHostComplexArray;		/* Host cufftComplex array */
cufftComplex* sDeviceComplexArray;		/* Device cufftComplex array */
cufftComplex* sDeviceComplexArrayTemp;	/* Temporary device cufftComplex array */

int sSizeComplex;	/* Size of complex arrays */
int sSizeFloat;		/* Size of single precisoin real arrays */


/********************************************************************
 * Name: 
 * CUDA_Way
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
extern "C"
void CUDA_Way()
{
	
	/* Sizes of the complex and real data "in bytes" used for 
	 * memory allocation */
	sSizeComplex = sizeof(cufftComplex) * sDataSize * sDataSize * sDataSize; 
	sSizeFloat = sizeof(float) * sDataSize * sDataSize * sDataSize; 

	/* Allocation of the complex array on the host side */
	sHostComplexArray = (cufftComplex*) malloc(sSizeComplex);

	/* Filling the spatial volume in the real part of the complex 
	 * array */   
	printf("	Packing Volume Array for Single Precision Format ... \n"); 
	for (int i = 0; i < mVolumeSize; i++)
	{
		sHostComplexArray[i].x = mExtractedVolumeData[i]; 
		sHostComplexArray[i].y = 0;
	}
	printf("	Packing Done Successfully \n\n");

	/* Allocation of the complex arrays on the device side */
	cudaMalloc((void**)(&sDeviceComplexArray), sSizeComplex);
	cudaMalloc((void**)(&sDeviceComplexArrayTemp), sSizeComplex);
	
	/* */ 
	cutilSafeCall(cudaMalloc((void**)(&sDevice1DFloatArray), sSizeFloat * 2));

	/* Uploading the complex array from the CPU to the GPU */
	cutilSafeCall(cudaMemcpy(sDeviceComplexArray, sHostComplexArray, sSizeComplex, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(sDeviceComplexArrayTemp, sHostComplexArray, sSizeComplex, cudaMemcpyHostToDevice));
	 
    /* Reservation - Linking PBO to CUDA Context */ 
    // cutilSafeCall(cudaGraphicsMapResources(1, &volTextureResource, 0));    
    
    /* Size (in bytes) for reconstructed image at which PBO size 
	 * shall be allocated */
	size_t Size3DBytes = sDataSize * sDataSize * sDataSize * 2 * sizeof(float); 
   
	// Now I have the Data on the GPU Side, Let's Do It
	// Sne a pointer to the device complex array 
	LaunchVolumeProcessingCUDA(sDeviceComplexArray, sDeviceComplexArrayTemp, sDataSize, sDevice1DFloatArray);
	
	cutilSafeCall(cudaMemcpy(mTextureArray, sDevice1DFloatArray,Size3DBytes, cudaMemcpyDeviceToHost));

	
	/*
	printf("fsddf \n"); 
	for (int i = (sDataSize * sDataSize * sDataSize) - 100; i < sDataSize * sDataSize * sDataSize; i++)
		printf("%f \n", mTextureArray[i] ); 
	printf("------------------------------------------------ \n" );
	 */ 
	 


	 
 
	



} 