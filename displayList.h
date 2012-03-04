/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: displayList.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: Display list contains the coordinates of the 
				  interecting polygon with the 3D spectral texture to 
				  extract the projection-slice (k-space 2D slice).  
 * Note(s)		: We are just interested in the central slice (Z = 0)
 *********************************************************************/ 

#include "fourierVolumeRenderer.h"

/********************************************************************
 * Name: 
 * SetDisplayList
 *
 * Description: 
 * This function creates the central polygon (QUAD) that will intersect 
 * with the 3D spectral slice to get the projection-slice and updates a 
 * vertex array with the result. 
 * Attributes: Center (Z = 0), Side Lenght = 1. 
 *			   Just 1 Slice, 4 Vertices, 3 Coordinates.
 *
 * Formal Parameters:
 * void. 
 *
 * Returns:
 * void. 
 ********************************************************************/
void SetDisplayList(void)
{
	printf("Creating Display List ... \n"); 
	
	/* Central slice */ 
	float mCenter		= 0;
	
	/* Left & right sides */  
	float mSide			= 0.5;
	
	/* Central slice */ 
	int nSlices			= 1;  
	
	/* Number of verticies */ 
	int nElements		= 4 * nSlices; 
	
	/* Coordinates */ 
	GLfloat *vPoints	= new GLfloat [3 * nElements]; 
	GLfloat *ptr		= vPoints;

	/* Fill the Display List with Vertecies */
	*(ptr++) = -mSide;
	*(ptr++) = -mSide;
	*(ptr++) =  mCenter;

	*(ptr++) =  mSide;
	*(ptr++) = -mSide;
	*(ptr++) =  mCenter;

	*(ptr++) =  mSide;
	*(ptr++) =  mSide;
	*(ptr++) =  mCenter;

	*(ptr++) = -mSide;
	*(ptr++) =  mSide;
	*(ptr++) =  mCenter;

	/* Fill the Display List (VERTEX_ARRAY) with Vertecies */ 
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vPoints);
	mDiaplayList = glGenLists(1);
	glNewList(mDiaplayList, GL_COMPILE);
	glDrawArrays(GL_QUADS, 0, nElements); 
	glEndList();
	delete [] vPoints;
	
	printf("	Display List Created Successfully \n\n"); 
}
