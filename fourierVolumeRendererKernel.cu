/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fourierVolumeRendererKernel.cu
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/

/* Utilities & System Includes */ 
// #include <shrUtils.h>
#include <cutil_inline.h>


/* GPU profiling */ 
// #define GPU_PROFILING

/* */
texture < float2, 2, cudaReadModeElementType > inTex;

/* cuda FFT Plan */
cufftHandle fftPlan;

/********************************************************************
 * Name: 
 * __device__ GetPixelFloat
 *
 * Description: 
 * Returns the complex value of the 2D spectal slice texture at (x,y).   
 *
 * Formal Parameters:
 * x (int), y (int): Texel position in the input texture. 
 *
 * Returns:
 * valTex (float2) : Complex value of the textule at (x,y). 
 * 
 * Note(s):
 * This function executes on device and callable only from the device.
 * inTex is global.
 ********************************************************************/ 
__device__ 
float2 GetPixelFloat(int x, int y)
{
	float2 valTex = tex2D(inTex, x, y);
	return valTex;
}

/********************************************************************
 * Name: 
 * __global__ fftShift2D_Kernel
 *
 * Description: 
 * Wrapping around a 2D input array.   
 *
 * Formal Parameters:
 * devArrayOutput (float*) : Output shifted array, 
 * devArrayInput (float*) : Input array, 
 * arrSize1D (int) : Size of the projection-slice in 1D. 
 *
 * Returns:
 * void. 
 * 
 * Note(s):
 * This function executes on device and callable from the host.
 ********************************************************************/
__global__
void fftShift2D_Kernel(float* devArrayOutput, float* devArrayInput, int arrSize1D)
{
	/* 1D Line */  
	int sLine = arrSize1D; 
	
	/* 2D Slice */
	int sSlice = arrSize1D * arrSize1D; 
	
	/* Transformations Equations */
	int sEq1 = (sSlice + sLine) / 2;
	int sEq2 = (sSlice - sLine) / 2; 
	
	/* Thread index (1D) */
	int xThreadIdx = threadIdx.x;
	int yThreadIdx = threadIdx.y;

	/* Block width & height */
	int blockWidth = blockDim.x;
	int blockHeight = blockDim.y;

	/* Thread index (2D) */  
	int xIndex = blockIdx.x * blockWidth + xThreadIdx;
	int yIndex = blockIdx.y * blockHeight + yThreadIdx;
	
	/* Thread (x,y) index converted into 1D index */
	int index = (yIndex * arrSize1D) + xIndex;
	
	if (xIndex < arrSize1D / 2)
	{
		if (yIndex < arrSize1D / 2)
		{
			/* First Quad */
			devArrayOutput[index] = devArrayInput[index + sEq1]; 
		}
		else 
		{
			/* Third Quad */
			devArrayOutput[index] = devArrayInput[index - sEq2]; 
		}
	}
	else 
	{
		if (yIndex < arrSize1D / 2)
		{
			/* Second Quad */
			devArrayOutput[index] = devArrayInput[index + sEq2];
		}
		else 
		{
			/* Fourth Quad */
			devArrayOutput[index] = devArrayInput[index - sEq1]; 
		}
	}
}

/********************************************************************
 * Name: 
 * __global__ TransferData
 *
 * Description: 
 * Transfering projection-slice data from into texture array.     
 *
 * Formal Parameters:
 * devArrayOutput (float*) : Output shifted array, 
 * devArrayInput (float*) : Input array, 
 * arrSize1D (int) : Size of the projection-slice in 1D.  
 *
 * Returns:
 * void. 
 * 
 * Note(s):
 ********************************************************************/ 
__global__
void TransferData(cufftComplex* fftResult, float* outputTextureImage, float* TempArray, int imgWidth, int imgHeight, int fScaleFactor)
{
	/* Thread ID */
	int xThreadID = threadIdx.x;
	int yThreadID = threadIdx.y;

	/* Block width & height */
	int blockWidth = blockDim.x;
	int blockHeight = blockDim.y;

	/* Thread index 2D */  
	int xIndex = blockIdx.x * blockWidth + xThreadID;
	int yIndex = blockIdx.y * blockHeight + yThreadID;
	
	/* Thread Index Converted into 1D Index */
	int index = ((yIndex * imgWidth) + xIndex); 
	
	float realVal = fftResult[index].x / (imgWidth * imgHeight * fScaleFactor); 
	float imgVal  = fftResult[index].y / (imgWidth * imgHeight * fScaleFactor);
	
	/* Save the complex value to the output texture array */ 
 	float varTemp = sqrt((realVal * realVal) + (imgVal * imgVal)) / 1;
 	outputTextureImage[index] =  varTemp; 
 	
	/* Save the compex value to the TempArray to be used later ! */
	TempArray[index] = varTemp; 
 } 

/********************************************************************
 * Name: 
 * __global__ ProcessCUDA
 *
 * Description: 
 * Processing CUDA.     
 *
 * Formal Parameters:
 * imgWidth (int) : Reconstructed image width,
 * imgHeight (int) : Reconstructed image height, 
 * fftInput (cufftComplex*) : Input projection-slice to 2D iFFT.    
 *
 * Returns:
 * void. 
 * 
 * Note(s): 
 ********************************************************************/
__global__ 
void ProcessCUDA(int imgWidth, int imgHeight, cufftComplex* fftInput)
{
	/* Thread ID */
	int xThreadID = threadIdx.x;
	int yThreadID = threadIdx.y;

	/* Block width/height */
	int blockWidth = blockDim.x;
	int blockHeight = blockDim.y;

	/* Thread index */ 
	int x = blockIdx.x * blockWidth + xThreadID;
	int y = blockIdx.y * blockHeight + yThreadID;
	
	/* Fetching pixel value from input texture */ 
	float2 pixelVal = GetPixelFloat(x, y); 

	/* Save opixel in the complex array */   
	fftInput[(y * imgWidth) + x].x = pixelVal.x;
	fftInput[(y * imgWidth) + x].y = pixelVal.y;
}

/********************************************************************
 * Name: 
 * LaunchProcessingCUDA
 *
 * Description: 
 * Transfering projection-slice data from into texture array.     
 *
 * Formal Parameters:
 * fGrid (dim3) : CUDA grid , 
 * fBlock (dim3) : CUDA block , 
 * inputSpectSlice (cudaArray*) : Input spectral projection-slice, 
 * outputImage (float*) : Output reconstructed image, 
 * imgWidth (int) : Reconstructed image width, 
 * imgHeight (int) : Reconstructed image height,
 * fftInput (cufftComplex*) : Input array to CUFFT, 
 * fftOutput (cufftComplex*) : Output array from the CUFFT, 
 * TempArray (float*) : Temporary array used during the 2D FFT shift, 
 * ScaleFactor (int) : Scaling factor to normalize the output image.  
 *
 * Returns:
 * void. 
 * 
 * Note(s): For "channle descriptor", see CUDA manual. 
 ********************************************************************/ 
extern "C" 
void LaunchProcessingCUDA ( dim3 fGrid, 
							dim3 fBlock, 
							cudaArray* inputSpectSlice, 
							float* outputImage, 
							int imgWidth, 
							int imgHeight, 
							cufftComplex* fftInput, 
							cufftComplex* fftOutput, 
							float* TempArray, 
							int ScaleFactor )
{
	/* Bind input projection-slice (CUDA Array) into inTex Texture */
	cutilSafeCall(cudaBindTextureToArray(inTex, inputSpectSlice));

	/* Get texture format (or channle descriptor) */ 
	struct cudaChannelFormatDesc formatTexture; 
	cutilSafeCall(cudaGetChannelDesc(&formatTexture, inputSpectSlice));

#ifdef GPU_PROFILING
	/* Profiling timer */
	unsigned int timerGPU = 0;
	
	/* Creating timer */
	cutilCheckError(cutCreateTimer(&timerGPU));
	
	/* Number of iterations */
	int numIterations = 30;
	for(int i = -1; i < numIterations; ++i) 
	{
		if(i == 0)
		cutilCheckError(cutStartTimer(timerGPU));
#endif  
		/* Resample input slice on CUDA */ 
		ProcessCUDA <<< fGrid, fBlock >>> (imgWidth, imgHeight, fftInput); 
	
		/* Doing inverse 2D cuFFT */
		cufftPlan2d(&fftPlan, imgWidth, imgHeight, CUFFT_C2C);
		cufftExecC2C(fftPlan, fftInput, fftOutput, CUFFT_INVERSE);

		/* Getting real image & transfer data to texture */
		TransferData <<< fGrid, fBlock >>> (fftOutput, outputImage, TempArray, imgWidth, imgHeight, ScaleFactor); 
		
		dim3 dimBlock(4,4,1);
		dim3 dimGrid(imgWidth / 4, imgHeight / 4, 1); 

		/* Wrap-around */
		fftShift2D_Kernel <<< dimGrid, dimBlock >>> (outputImage, TempArray, imgWidth); 
			
#ifdef GPU_PROFILING
	}
	
	/* Synchronization */ 
	cudaThreadSynchronize();
	
	/* Stop timer */
	cutilCheckError(cutStopTimer(timerGPU));

	/* Profiling benchmarks */ 
	double secondsGPU = cutGetTimerValue(timerGPU) / ((double) numIterations * 1000.0);
	double numTexels = (double) imgWidth * (double) imgHeight;
	double numTexPerSec = 1.0e-6 * (numTexels / secondsGPU);
	
#endif
}





