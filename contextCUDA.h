/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: contextCUDA.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * InitCUDAContext
 *
 * Description: 
 * Initializing CUDA/OpenGL contexts and setting them to the device "GPU" .    
 *
 * Formal Parameters:
 * argc (int) : Number of arguments, 
 * argv (char **) : Arguments' strings, 
 * UseGL (bool) : Using OpenGL or not.  
 *
 * Returns:
 * (bool) : If correct initialization.  
 * 
 * Note(s):
 ********************************************************************/
bool InitCUDAContext(int argc, char **argv, bool UseGL)
{
    if (UseGL)
    {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
            cutilGLDeviceInit(argc, argv);
        else 
            cudaGLSetGLDevice (cutGetMaxGflopsDeviceId());
    } 
    else 
    {
        if (cutCheckCmdLineFlag(argc, (const char **)argv, "device"))
            cutilDeviceInit(argc, argv);
        else 
            cudaSetDevice (cutGetMaxGflopsDeviceId());
    }

	return true;
}

/********************************************************************
 * Name: 
 * TransferSliceToCUDA
 *
 * Description: 
 * Processing slice on CUDA.    
 *
 * Formal Parameters:
 * fImgWidth (int) : Projection slice width, 
 * fImgHeight (int) : Projection slice height. 
 *
 * Returns:
 * void.  
 * 
 * Note(s):
 ********************************************************************/
void TransferSliceToCUDA(int fImgWidth, int fImgHeight)
{
    /* Extracted projection-slice from 3D spectrum texture is sent to CUDA 
	 * for processing */
    cudaArray *fExtractedTextureSlice; 
    
    /* Reconstructed image after CUDA processing & inverse cuFFT 2D */ 
    float* fReconstructedImage;
    
    /* Reservation - Linking PBO to CUDA Context */ 
    cutilSafeCall(cudaGraphicsMapResources(1, &CUDA_PBODestinationBufferResource, 0));    
    
    /* Size (in bytes) for the reconstructed image at which PBO size 
	 * shall be allocated */ 
    size_t fSizeRecImageBytes; 
    
    /* Actual mapping - map the reconstructed image from CUDA context 
	 * to the PBO destination resource that be accessed by OpenGL to 
	 * render the texture */ 
    cutilSafeCall
    (cudaGraphicsResourceGetMappedPointer
    ((void **) &fReconstructedImage, &fSizeRecImageBytes, CUDA_PBODestinationBufferResource));

    /* Reservation - linking slice texture to CUDA context */ 
    /* Map buffer objects to get CUDA device pointers */ 
    cutilSafeCall
    (cudaGraphicsMapResources(1, &CUDASliceResourceBuffer, 0));
    
    /* Link extracted slice from OpenGL context to CUDA context */  
    cutilSafeCall
    (cudaGraphicsSubResourceGetMappedArray(&fExtractedTextureSlice, CUDASliceResourceBuffer, 0, 0));

    /* Block size */ 
    dim3 block(16, 16, 1);
    
    /* Grid size */ 
    dim3 grid(fImgWidth / block.x, fImgHeight / block.y, 1); 

    /* Execute CUDA kernel on the extratced projection-slice */ 
	LaunchProcessingCUDA(grid, block, fExtractedTextureSlice, fReconstructedImage, fImgWidth, fImgHeight, cuFFT_Input, cuFFT_Output, TempArray, mScaleFactor);
    
    /* Unlink all buffers connected to CUDA contexts to be used by OpenGL */ 
    cudaGraphicsUnmapResources(1, &CUDASliceResourceBuffer, 0);
    cudaGraphicsUnmapResources(1, &CUDA_PBODestinationBufferResource, 0);
}

/********************************************************************
 * Name: 
 * TransferSliceToCUDA
 *
 * Description: 
 * Copy image & process it on CUDA device .    
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void.  
 * 
 * Note(s):
 ********************************************************************/
// 
void ProcessSlice()
{
    /* Run the CUDA kernel */ 
    TransferSliceToCUDA(mImageWidth, mImageHeight);
	
    /* Bind image to PBO */ 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, PBODestinationBuffer);

    /* Bind texture image to buffer */ 
    glBindTexture(GL_TEXTURE_2D, ImgTextureID);

    // Replace the old image with the new reconstructed one
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mImageWidth, mImageHeight, GL_LUMINANCE, GL_FLOAT, NULL);

    // Ceck for OpenGL/CUDA Errors
    // CUT_CHECK_ERROR_GL();

    /* Release buffers */ 
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

