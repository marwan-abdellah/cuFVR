/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fftShift.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * FFT_Shift_2D
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
float** FFT_Shift_2D(float** iArr, float** oArr, int N)
{	  
	for (int i = 0; i < N/2; i++)
		for(int j = 0; j < N/2; j++)
		{
			oArr[(N/2) + i][(N/2) + j] = iArr[i][j];
			oArr[i][j] = iArr[(N/2) + i][(N/2) + j];
			
			oArr[i][(N/2) + j] = iArr[(N/2) + i][j];
			oArr[(N/2) + i][j] = iArr[i][(N/2) + j];
		}
	
	return oArr;
}

/********************************************************************
 * Name: 
 * Repack_2D
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
float* Repack_2D(float** Input_2D, float* Input_1D, int N)
{
	int ctr = 0; 
	for (int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
		{
			Input_1D[ctr] = Input_2D[i][j];
			ctr++;
		}
	
	return Input_1D;
}

/********************************************************************
 * Name: 
 * FFT_Shift_3D
 *
 * Description: 
 * Loading the raw volume and returning a pointer to the containing 
 * array. 
 *
 * Formal Parameters:
 * Input (float*) : Input 1D array, 
 * N : Each dimension in the array. 
 *
 * Returns:
 * oArr (float***) : 3D array for the shifted volume. 
 * 
 * Note(s):
 * Number of elemnts in the array is (N * N * N), ctr is internal. 
 ********************************************************************/
float*** FFT_Shift_3D(float* Input, int N)
{
	/* 3D input and output arrays */
	float ***iArr, ***oArr;;
	
	/* Alloacting iArr (3D) */
	iArr = (float***) malloc(N * sizeof(float**));
	for (int y = 0; y < N; y++)
	{
		iArr[y] = (float**) malloc (N * sizeof(float*));
		for (int x = 0; x < N; x++)
		{
			iArr[y][x] = (float*) malloc( N * sizeof(float));
		}
	}
	
	/* Allocating oArr (3D) */
	oArr = (float***) malloc(N * sizeof(float**));
	for (int y = 0; y < N; y++)
	{
		oArr[y] = (float**) malloc (N * sizeof(float*));
		for (int x = 0; x < N; x++)
		{
			oArr[y][x] = (float*) malloc( N * sizeof(float));
		}
	}
	
	/* Moving the input 1D array into the 3D iArr array */
	int ctr = 0; 
	for (int k = 0; k < N; k++)
		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				iArr[i][j][k] = Input[ctr];
				oArr[i][j][k] = 0;
				ctr++;
			}
	
	/* Doing the 3D fft shift operation */
	for (int k = 0; k < N/2; k++)
		for (int i = 0; i < N/2; i++)
			for(int j = 0; j < N/2; j++)
			{
				oArr[(N/2) + i][(N/2) + j][(N/2) + k] = iArr[i][j][k];
				oArr[i][j][k] = iArr[(N/2) + i][(N/2) + j][(N/2) + k];
				
				oArr[(N/2) + i][j][(N/2) + k] = iArr[i][(N/2) + j][k];
				oArr[i][(N/2) + j][k] = iArr[(N/2) + i][j][(N/2) + k];
				
				oArr[i][(N/2) + j][(N/2) + k] = iArr[(N/2) + i][j][k]; 
				oArr[(N/2) + i][j][k] = iArr[i][(N/2) + j][(N/2) + k]; 
				
				
				oArr[i][j][(N/2) + k] = iArr[(N/2) + i][(N/2) + j][k];
				oArr[(N/2) + i][(N/2) + j][k] = iArr[i][j][(N/2) + k]; 
			}
	
	/* Dellocating iArr */
	delete [] iArr;
	return oArr;
}

/********************************************************************
 * Name: 
 * Repack_3D
 *
 * Description: 
 * Reapcking 3D array contetnts into 1D array.  
 *
 * Formal Parameters:
 * Input_3D (float***) : Input 3D array,  
 * Input_1D (float* ) : Input 1D array, 
 * N : Number of elements in each dimension.   
 *
 * Returns:
 * Input_1D (float*) : Pointer to the 1D array containg the volume. 
 * 
 * Note(s):
 * ctr is internal.  
 ********************************************************************/
float* Repack_3D(float*** Input_3D, float* Input_1D, int N)
{
	/* Repacking the 3D volume into 1D array */
	int ctr = 0; 
	for (int k = 0; k < N; k++)
		for (int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
			{
				Input_1D[ctr] = Input_3D[i][j][k];
				ctr++;
			}
	
	return Input_1D;
}


