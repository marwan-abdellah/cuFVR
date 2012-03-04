/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: cuda3DKernel.cu
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: This module handles volume processing on the device. 
 * Note(s)		: 
 *********************************************************************/ 

/* CUDA utilities & system includes */ 
/* #include <shrUtils.h> */
#include <cutil_inline.h>

/** Kernel globals **/

/* CUDA FFT plan */
cufftHandle fftPlan3D;

/* Host array to validate GPU implementation results */
cufftComplex* CPU_ARRAY; 

/********************************************************************
 * Name: 
 * __global__ fftShift3D_cufftComplex_i
 *
 * Description: 
 * Loading the raw volume and returning a pointer to the containing 
 * array.
 * 
 * Formal Parameters:
 * devArrayInput (cufftComplex*) : Input array, 
 * devArrayOutput (cufftComplex*) : Output array, 
 * arrSize1D (int) : Size of the volume in 1D assumming unified 
					 dimensionality, 
 * zIndex (int) : To complete indexing an element in the volume. 
 *
 * Returns:
 * void.  
 * 
 * Note(s):
 * This kernel executes in coordination with the function fftShift3D_i. 
 * This kernel executes on the device and callable from the host.
 * "zIndex" does all the magic, try it yourself to believe it.  
 ********************************************************************/
__global__
void fftShift3D_cufftComplex_i( cufftComplex* devArrayInput, 
								cufftComplex* devArrayOutput, 
								int arrSize1D, 
								int zIndex )
{
	/* 3D volume & 2D slice & 1D line */ 
	int sLine = arrSize1D; 
	int sSlice = arrSize1D * arrSize1D; 
	int sVolume = arrSize1D * arrSize1D * arrSize1D; 
	
	/* Transformation equations */ 
	int sEq1 = ( sVolume + sSlice + sLine ) / 2;
	int sEq2 = ( sVolume + sSlice - sLine ) / 2; 
	int sEq3 = ( sVolume - sSlice + sLine ) / 2;
	int sEq4 = ( sVolume - sSlice - sLine ) / 2; 
	
	/* Thread index */
	int xThreadIdx = threadIdx.x;
	int yThreadIdx = threadIdx.y;

	/* Block width & height */
	int blockWidth = blockDim.x;
	int blockHeight = blockDim.y;

	/* Thread index 2D */  
	int xIndex = blockIdx.x * blockWidth + xThreadIdx;
	int yIndex = blockIdx.y * blockHeight + yThreadIdx;
	
	// Thread index converted into 1D index
	int index = ( zIndex * sSlice ) + ( yIndex * sLine ) + xIndex;
	
	if ( zIndex < arrSize1D / 2 )
	{
		if ( xIndex < arrSize1D / 2 )
		{
			if ( yIndex < arrSize1D / 2 )
			{
				/* First Quad */
				devArrayOutput[index] = devArrayInput[index + sEq1]; 
			}
			else 
			{
				/* Third Quad */ 
				devArrayOutput[index] = devArrayInput[index + sEq3]; 
			}
		}
		else 
		{
			if ( yIndex < arrSize1D / 2 )
			{
				/* Second Quad */ 
				devArrayOutput[index] = devArrayInput[index + sEq2];
			}
			else 
			{
				/* Fourth Quad */
				devArrayOutput[index] = devArrayInput[index + sEq4]; 
			}
		}
	}
	else 
	{
		if ( xIndex < arrSize1D / 2 )
		{
			if ( yIndex < arrSize1D / 2 )
			{
				/* First Quad */ 
				devArrayOutput[index] = devArrayInput[index - sEq4]; 
			}
			else 
			{
				/* Third Quad */ 
				devArrayOutput[index] = devArrayInput[index - sEq2]; 
			}
		}
		else 
		{
			if ( yIndex < arrSize1D / 2 )
			{
				/* Second Quad */ 
				devArrayOutput[index] = devArrayInput[index - sEq3];
			}
			else 
			{
				/* Fourth Quad */
				devArrayOutput[index] = devArrayInput[index - sEq1]; 
			}
		}
	}
}

/********************************************************************
 * Name: 
 * fftShift3D_i
 *
 * Description: 
 * Loading the raw volume and returning a pointer to the containing 
 * array. 
 *
 * Formal Parameters:
 * _arrayDeviceInput (cufftComplex*) : Input array, 
 * _arrayDeviceOutput (cufftComplex*) : Output array, 
 * _arraySize1D (int) : Size of the volume in 1D assumming unified 
						dimensionality, 
 * _block (dim3) : CUDA block configuration,
 * _grid (dim3) : CUDA grid configuration. 
 
 * Returns:
 * void. 
 * 
 * Note(s):
 * This function is callable from the CPU and executes on the device.
 * This function is to be treated exactly as CUDA __global__ one.
 * This function executes in coordination with the kernel 
   fftShift3D_cufftComplex_i.  
 ********************************************************************/
void fftShift3D_i( cufftComplex* _arrayDeviceInput, 
				   cufftComplex* _arrayDeviceOutput, 
				   int _arraySize1D, 
				   dim3 _block, 
				   dim3 _grid )
{
	for ( int i = 0; i < _arraySize1D; i++ )
		fftShift3D_cufftComplex_i <<< _grid, _block >>> 
		( _arrayDeviceInput, _arrayDeviceOutput, _arraySize1D, i );
}

/********************************************************************
 * Name: 
 * __global__ Pack1DComplexArray
 *
 * Description: 
 * Packing complex array of 2 float compoenets into 1D float sorted in 
 * consecutive fashion, i.e. real value at odd indicies and imaginary 
 * values at even indicies. 
 *
 * Formal Parameters:
 * complexArray (cufftComplex*) : Input complex array, 
 * array1D (float*) : Resulting float array, 
 * arrSize1D (int) : Size of the volume in 1D assumming unified 
 *					 dimensionality.
 *
 * Returns:
 * void. 
 * 
 * Note(s):
 ********************************************************************/
__global__
void Pack1DComplexArray( cufftComplex* complexArray, 
						 float* array1D, 
						 int arrSize1D )
{
	// Index (1D) 
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i ndex < ( arrSize1D * arrSize1D * arrSize1D ) )
	{
		/* Real value */
		array1D[(2 * index)    ]	= complexArray[index].x;
		
		/* Imaginary value */
		array1D[(2 * index) + 1]	= complexArray[index].y; 
	}

}

/********************************************************************
 * Name: 
 * LaunchVolumeProcessingCUDA
 *
 * Description: 
 * Volume processing on the GPU that includes shifting the spatial and
 * the spectral volumes, doing the 3D FFT with cuFFT, repacking the 
 * resulting shifted spectral volume into 1D array sto suit a 2-component 
 * (RG) OpenGL 3D texture.     
 *
 * Formal Parameters:
 * MainComplexArray (cufftComplex*) : Main complex array residing on 
									  the GPU, 
 * TempComplexArray (cufftComplex*) : Temporary complex array used on 
									  the fly, 
 * fDataSize (int) : Size of the volume in 1D assumming unified 
 *					 dimensionality, 
 * ptrTo1DArray (float*) : OpenGL-compatiable array containg the shifted 
						   spectral volume. 
 *
 * Returns:
 * void. 
 * 
 * Note(s):
 * Some testing code is commented to reduce their overhead. Uncomment 
 * them if needed. 
 ********************************************************************/
extern "C" 
void LaunchVolumeProcessingCUDA( cufftComplex* MainComplexArray, 
								 cufftComplex* TempComplexArray, 
								 int fDataSize, 
								 float* ptrTo1DArray )
{
	/* In this point, I have a pointer to the complex array residing 
	* in the GPU memory. This complex array will carry the spatial 
	* and the spectral volumes alternatively */
	
	/* CUDA volume processing configuration */
	dim3 sBlock(16, 16, 1);
	dim3 sGrid((fDataSize / 16), (fDataSize / 16), 1);

	/* Wrapping-around the spatial volume */
	fftShift3D_i(TempComplexArray,  MainComplexArray, fDataSize, sBlock, sGrid);
	
	/* Allocating a CPU array to receive resulting volumes on the GPU
	 * for validation */
	CPU_ARRAY = (cufftComplex*) malloc 
			(fDataSize * fDataSize * fDataSize * sizeof(cufftComplex));
			 
	/* Copy the final volume from the device to the host and test it */
	cutilSafeCall(
	cudaMemcpy( CPU_ARRAY, MainComplexArray, 
				fDataSize * fDataSize * fDataSize * sizeof( cufftComplex ), 
				cudaMemcpyDeviceToHost ) );
	
	/* Testing the last 100 elements of the spatial volume on the GPU */  
	/* 
	printf("Testing the spatial volume before doing the cuFFT \n"); 
	for (int i = ( fDataSize * fDataSize * fDataSize) - 100; i < fDataSize * fDataSize * fDataSize; i++ )
		printf("%f \n", CPU_ARRAY[i].x); 
	*/
	 
	/* Setup for 3D cuFFT plan */
	cufftPlan3d(&fftPlan3D, fDataSize, fDataSize, fDataSize, CUFFT_C2C);
	cufftExecC2C(fftPlan3D, MainComplexArray, TempComplexArray, CUFFT_FORWARD);
	
	/* Copy the resulting spectrum from the cuFFT operation to the host 
	 * and test it */
	/* cutilSafeCall(cudaMemcpy(CPU_ARRAY, 
					 TempComplexArray, 
					 (fDataSize * fDataSize * fDataSize * sizeof(cufftComplex)), 
					 cudaMemcpyDeviceToHost)); */ 
	
	/* Testing the last 100 elements of the spectral volume on the GPU */
	/*
	printf("Testing the resulting spectral volume after doing the cuFFT \n"); 
	for (int i = ( fDataSize * fDataSize * fDataSize) - 100; i < fDataSize * fDataSize * fDataSize; i++ )
		printf("%f \n", CPU_ARRAY[i].x); 
	*/
	 
	// Do 3D FFT Shift for the Generated Spectrum
	// Save the Output in the Temp Array
	// fftShift3D_cufftComplex <<< sGrid, sBlock >>>(TempComplexArray, MainComplexArray, fDataSize);
	
	/* Wrapping-around the spectral volume to center its DC component */
	fftShift3D_i( TempComplexArray, MainComplexArray, fDataSize, sBlock, sGrid );

	/* Copy the resulting shifted spectrum from the "fftShift3D_i operation" to 
	 * the host and test it */
	/* cutilSafeCall(cudaMemcpy(CPU_ARRAY, 
					 TempComplexArray, 
					 (fDataSize * fDataSize * fDataSize * sizeof(cufftComplex)), 
					 cudaMemcpyDeviceToHost)); */ 
	
	/* Testing the last 100 elements of the shifted spectral volume on the GPU */
	/*
	printf("Testing the resulting shifted spectral volume after wrapping it around \n"); 
	for (int i = ( fDataSize * fDataSize * fDataSize) - 100; i < fDataSize * fDataSize * fDataSize; i++ )
		printf("%f \n", CPU_ARRAY[i].x); 
	*/
	
	/* Garbbage */  
	/*
	cutilSafeCall(cudaMemcpy(CPU_ARRAY, MainComplexArray, 128 * 128 * 128 * sizeof(cufftComplex), cudaMemcpyDeviceToHost)); */
	printf("Testing for a specific volume of 128 * 128 * 128 \n"); 
	for (int i = 0; i < (128 * 128 * 128 / 2); i++)
		printf("%f \n", CPU_ARRAY[i].x);
	*/

	/* Now, I have the spectral volume in cufftComlpex array. Save this 
	 * spectrum to 1D array on the device & link this array to OpenGL Via 
	 * cudaGraphics Resource */ 

	/* GPU configuration for the packing kernel */
	dim3 sBlock1D( 512,1,1 ); 
	dim3 sGrid1D( ( fDataSize * fDataSize * fDataSize ) / 512, 1, 1 ); 

	/* Run 1D packing for texture array */
	Pack1DComplexArray <<< sGrid1D, sBlock1D >>> ( MainComplexArray, ptrTo1DArray, fDataSize );   
}
