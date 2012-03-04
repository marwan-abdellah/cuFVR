/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering, 
 * Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fourierVolumeRenderer.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: Main header file for the project. 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"
void InitVars()
{
	printf("-------------------------- \n"); 
	printf("Initializing Variables ... \n");
	printf("-------------------------- \n\n"); 
	


	// Volume Attributes
	mVolWidth_DS			= 256;
	mVolHeight_DS			= 256;
	mVolDepth_DS			= 256;

	mVolumeSize_DS 		= mVolWidth_DS * mVolHeight_DS * mVolDepth_DS;
	mVolumeSizeBytes_DS	= mVolumeSize_DS * sizeof(mVolumeType);

	
	// Volume Attributes
	mVolWidth			= 256;
	mVolHeight			= 256;
	mVolDepth			= 256;
	mUniDim 			= 256;

	mVolArea 			= mVolWidth * mVolHeight;
	mVolumeSize 		= mVolWidth * mVolHeight * mVolDepth;
	mVolumeSizeBytes	= mVolumeSize * sizeof(mVolumeType);
	
	printf("Loaded Volume Attributes  \n\n"); 
	printf("    Volume Width        : [%d] \n", mVolWidth);
	printf("    Volume Height       : [%d] \n", mVolHeight);
	printf("    Volume Depth        : [%d] \n", mVolDepth);
	printf("    Volume Unified Dim. : [%d] \n", mUniDim);
	printf("    Volume Size         : [%d] x [%d] x [%d] = [%d] Voxels\n", mVolWidth, mVolHeight, mVolDepth, mVolumeSize);
	printf("    Volume Cross-Sec.   : [%d] x [%d] = [%d] Pixels  \n\n", mVolWidth, mVolHeight, mVolArea);
	
	printf("    Volume Size (Bytes)    : [%d] Bytes \n", mVolumeSizeBytes);
	if ((mVolumeSizeBytes / (1024)) != 0)
	printf("    Volume Size (kBytes)   : [%d] kBytes \n", mVolumeSizeBytes / 1024);
	if ((mVolumeSizeBytes / (1024 * 1024)) != 0)
	printf("    Volume Size (MBytes)   : [%d] MBytes \n\n", mVolumeSizeBytes / (1024 * 1024));
	
	printf("    Complex 3D Spectrum Size (kBytes) : [%d] kBytes \n", mVolumeSizeBytes  * (sizeof(float) * 8 ) * 2 / (1024));
	if ((mVolumeSizeBytes  * (sizeof(float) * 8 ) * 2 / (1024 * 1024)) != 0)
	printf("    Complex 3D Spectrum Size (MBytes) : [%d] MBytes \n\n", mVolumeSizeBytes  * (sizeof(float) * 8 ) * 2 / (1024 * 1024));
	printf("------------------------------------------------------------------------------ \n\n"); 
	
	// Slice & Image Attributes
	mSliceWidth 	= mVolWidth; 
	mSliceHeight 	= mVolHeight;
	mSliceSize 		= mSliceWidth * mSliceHeight;

	mImageWidth 	= mVolWidth;
	mImageHeight	= mVolHeight;
	
	printf("Extracted Slice & Reconstructed Image Attributes  \n\n"); 
	printf("    (Slice / Image) Width    : [%d] \n", mSliceWidth);
	printf("    (Slice / Image) Height   : [%d] \n", mSliceHeight);
	printf("    (Slice / Image) Area     : [%d] x [%d] = [%d] Pixels / Texels \n\n", mSliceWidth, mSliceHeight, mSliceSize);
	
	printf("    Complex Slice Size (Bytes)  : [%d] Bytes \n", mSliceSize * (sizeof(float) * 8) * 2);
	if ((mSliceSize * (sizeof(float) * 8) * 2 / 1024) != 0)
	printf("    Complex Slice Size (kBytes) : [%d] kBytes \n\n", mSliceSize * (sizeof(float) * 8) * 2 / 1024);
	
	printf("    Image Size (Bytes)  : [%d] Bytes \n", mSliceSize * (sizeof(float) * 8));
	printf("    Image Size (kBytes) : [%d] kBytes \n\n", mSliceSize * (sizeof(float) * 8) / 1024);
	printf("------------------------------------------------------------------------------ \n\n"); 

	// Texture Attributes
	mNumTexels 			= mSliceSize;
	mNumValues 			= mSliceSize * 1;
	mTextureDataSize 	= mNumValues * sizeof(float);
	
	printf("Texture Parameters  \n"); 
	printf("	Number of Texels in Slice	: %d \n", mNumTexels);
	printf("	Number of Color Components  	: %d \n", mNumValues);
	printf("	Texture Data Size 		: %d \n \n", mTextureDataSize); 
}

