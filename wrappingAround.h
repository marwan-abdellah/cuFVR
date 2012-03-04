/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: wrappingAround.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * WrapAroundVolume
 *
 * Description: 
 * Wrapping around or FFT-SHIFTing the "spatial" volume.    
 *
 * Formal Parameters:
 * void.  
 *
 * Returns:
 * void.  
 * 
 * Note(s):
 * mVol_3D and mVolumeDataFloat are globals. 
 ********************************************************************/ 
void WrapAroundVolume()
{
	printf("Wrapping Around Spatial Volume ... \n"); 
	
	/* Shifting the 3D input volume */
	mVol_3D = FFT_Shift_3D(mVolumeDataFloat, mUniDim);
	
	/* Repacking the resulting 3D array into 1D array */
	mVolumeDataFloat = Repack_3D(mVol_3D, mVolumeDataFloat, mUniDim);
	
	printf("	Wrapping Around Spatial Volume Done Successfully \n\n");
}

/********************************************************************
 * Name: 
 * WrapAroundSpectrum
 *
 * Description: 
 * Wrapping around or FFT-SHIFTing the "spectral" volume.    
 *
 * Formal Parameters:
 * void.  
 *
 * Returns:
 * void.  
 * 
 * Note(s):
 * mVolumeDataFloat, mVolumeArrayComplex, mVol_3D are globals.  
 ********************************************************************/ 
void WrapAroundSpectrum()
{
	printf("Wrapping Around Spectrum Data ... \n");
	
	/* Wrapping around the "real" part of the spectrum */
	printf("	Real Part .... \n");
	for (int i = 0; i < mVolumeSize; i++)
		mVolumeDataFloat[i] =  mVolumeArrayComplex[i][0]; 
	
	mVol_3D = FFT_Shift_3D(mVolumeDataFloat, mUniDim);
	mVolumeDataFloat = Repack_3D(mVol_3D, mVolumeDataFloat, mUniDim);
	
	for (int i = 0; i < mVolumeSize; i++)
		mVolumeArrayComplex[i][0] = mVolumeDataFloat[i]; 
	
	/* Wrapping around the "imaginary" part of the spectrum */
	printf("	Imaginary Part .... \n");
	
	for (int i = 0; i < mVolumeSize; i++)
		mVolumeDataFloat[i] =  mVolumeArrayComplex[i][1]; 
	
	mVol_3D = FFT_Shift_3D(mVolumeDataFloat, mUniDim);
	mVolumeDataFloat = Repack_3D(mVol_3D, mVolumeDataFloat, mUniDim);
	
	for (int i = 0; i < mVolumeSize; i++)
		mVolumeArrayComplex[i][1] = mVolumeDataFloat[i];
	
	printf("	Wrapping Around Spectrum Data Done Successfully \n\n");
	
	
} 

