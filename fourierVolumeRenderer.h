/********************************************************************* 
 * Copyrights (c) Marwan Abdellah. All rights reserved.  
 * This code is part of my Master's Thesis Project entitled "High 
 * Performance Fourier Volume Rendering on Graphics Processing Units 
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University. 
 * Please, don't use or distribute without authors' permission. 
 
 * File			: fourierVoumeRenderer.h.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com> 
 * Created		: April 2011
 * Description	: 
 * Note(s)		: 
 *********************************************************************/  
 
#ifndef _FOURIER_VOLUME_RENDERER_H
#define _FOURIER_VOLUME_RENDERER_H

#ifdef _WIN32
	#define WINDOWS_LEAN_AND_MEAN
	#define NOMINMAX
	#include <windows.h>
	#pragma warning(disable:4996)
#endif

/***************************************
 * OpenGL includes
 ***************************************/
#include <GL/glew.h>
#include <GL/glu.h>

#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
	#define USE_TEXSUBIMAGE2D
#else
	#include <GL/glut.h>
#endif

/***************************************
 * Utilities & System includes
 ***************************************/
#ifdef CUDA_SHR
	#include <shrUtils.h>
#endif 
#include <cutil_inline.h>

/***************************************
 * CUDA / OpenGL includes
 ***************************************/
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cudaGL.h>

/***************************************
 * FFTW (3) Library includes
 ***************************************/ 
#include "fftw3.h"

/***************************************
 * CUDA FFT includes 
 ***************************************/
#include "cufft.h"

// Shall be Removed Later ! 
#define USE_TEXSUBIMAGE2D

/***************************************
 * OpenGL Checking Macros 
 ***************************************/
#ifdef WIN32
     bool IsOpenGLAvailable(const char *appName) { return true; }
#else
  #if (defined(__APPLE__) || defined(MACOSX))
     bool IsOpenGLAvailable(const char *appName) { return true; }
  #else
     // Linux Machine
     #include <X11/Xlib.h>

     bool IsOpenGLAvailable(const char *appName)
     {
        Display *XDisplayGL = XOpenDisplay(NULL);
        if (XDisplayGL == NULL)
           return false;
        else
        {
           XCloseDisplay(XDisplayGL);
           printf("OpenGL Diaply is Assigned");  
           return true;
        }
     }
  #endif
#endif

/***************************************
 * Type Definitions  
 ***************************************/
typedef unsigned int 	uint;
typedef unsigned char 	uchar;
typedef unsigned char 	mVolumeType;  

/***************************************
 * Globals   
 ***************************************/

bool EnableCUDA = true;

float* sDevice1DFloatArray; 

float* TempArray; 


/** FFT globals **/

cufftComplex *cuFFT_Input;		/* Input array to cuFFT */
cufftComplex *cuFFT_Output;		/* Output array from cuFFT */

/** OpenGL globals **/
/* Framebuffer object resource for the projection-slice */ 
GLuint FBOSourceBuffer;
struct cudaGraphicsResource *CUDASliceResourceBuffer;




 

// Resulting Image Texture Buffer for CUDA

GLuint ImgTextureID;			/* Reconstruced image testure ID */
GLuint PBODestinationBuffer;	/* PBO destination buffer ID */
struct cudaGraphicsResource *CUDA_PBODestinationBufferResource;	
struct cudaGraphicsResource *CUDA_ResultingTextureBufferResource;

// 3D Texture 
GLuint mVolTexureID;			// 3D Spectrum Texture ID
GLuint PBOVolumeBuffer; 
struct cudaGraphicsResource *volTextureBufferResource;
struct cudaGraphicsResource *volumePBOBufferResource; 

float* Texture3D_1DArrayPointer; 

/** GLUT Globals **/
int mWindowWidth			= 512;	/* Window width */
int mWindowHeight			= 512;	/* Window height */
unsigned int mImageWidth 	= 0;	/* Reconstructed image width */
unsigned int mImageHeight	= 0;	/* Reconstructed image height */
int iGLUTWindowHandle		= 0;	/* OpenGL window handle */



char* mPath = "F:/Microsoft Projects/Data/CTHead_1/CTData.img"; /* Volume file path */ 
const char *sSDKname = "fourierVolumeRenderer";		


/** Dimensions of the final volume **/    
int mVolWidth			= 0;	/* Volume width	*/
int mVolHeight			= 0;	/* Volume height */
int mVolDepth			= 0;	/* Volume depth	*/
int mUniDim 			= 0;	/* Volume Unified dimension in X, Y, and Z */
int mVolArea 			= 0;	/* Area of cross-section in the volume */
int mVolumeSize			= 0;	/* Size of the final volume = width x height x depth */
int mVolumeSizeBytes	= 0;	/* Size of the final volume in bytes */

/** Dimensions of the loaded volume dataset "_DS" **/  
int mVolWidth_DS		= 0;	/* Volume width	*/
int mVolHeight_DS		= 0;	/* Volume height */
int mVolDepth_DS		= 0;	/* Volume depth	*/
int mVolumeSize_DS		= 0;	/* Size of the loaded volume = width x height x depth */
int mVolumeSizeBytes_DS	= 0;	/* Size of the loaded volume in bytes */

char ***volIn3D, ***volOut3D;
char* mExtractedVolumeData; 

/** Projection-slice parameters **/
int mSliceWidth 		= 0;	/* projcetion-slice width */
int mSliceHeight 		= 0;	/* projection-slice height */
int mSliceSize			= 0;	/* projection-slice size = width x height */


int mScaleFactor		= 1000;



// Slice Attributes ________________________________________________________*/


// FFTW Globals ____________________________________________________________*/ 
fftwf_complex* 	mVolumeArrayComplex;
fftwf_complex* 	mSliceArrayComplex;
fftwf_plan 	mFFTWPlan;  

/** Wrapping Around Globals **/
float** 	mImg_2D; 
float** 	mImg_2D_Temp; 
float***	mVol_3D;  


// OpenGL Texture Arrays
float* 		mTextureArray;
float* 		mFrameBufferArray;

/* OpenGL projection-slice texture globals */ 
GLuint mSliceTextureID; 		/* Extracted slice ID */ 
GLuint mSliceTextureSrcID; 		/* Input texture to CUDA ID */ 
GLuint mSliceTextureResID;		/* Destination of CUDA results ID */ 
int mNumTexels 			= 0;	/* Number of textels in the texture slice */ 
int mNumValues 			= 0;	/* Twice as the number of texels (complex component) */ 
int mTextureDataSize 	= 0;	/* Size of the texture slice = width * height */ 
unsigned int mSizeTextureData;	/* Texture slice size in bytes */


GLuint FrameBufferID;			// FrameBuffer
GLuint SliceTextureID;			// Spectrum slice texture



/** Framebufer object targets **/
const GLenum fbo_targets[] = {  GL_COLOR_ATTACHMENT0_EXT, 
								GL_COLOR_ATTACHMENT1_EXT, 
								GL_COLOR_ATTACHMENT2_EXT, 
								GL_COLOR_ATTACHMENT3_EXT  };

/* Timer globals */
static int mFPSCount 	= 0;
static int mFPSLimit 	= 1;
unsigned int mTimer		= 0;

/** OpenGL globals **/ 
int mXrot				= 0;	/* Rotation around x-axis */
int mYrot				= 0;	/* Rotation around y-axis */
int mZrot				= 0;	/* Rotation around z-axis */
int mScalingFactor 		= 50;	/* Scaling factor */
GLuint mDiaplayList;			/* Display list ID */


/** Volume & image globals **/
char* 		mVolumeData;					/* Loaded input volume */
uchar*		mRecImage;						/* Reconstructed image */
float* 		mVolumeDataFloat;				/* Loaded volume in single precision */
float*	 	mAbsoluteReconstructedImage;	/* Absolute of the reconstructed image */

/***************************************
 * Forward declarations 
 ***************************************/
/* Loading volume & data initialization */
void InitVars(); 
void InitData(); 
char* LoadRawFile(char* fFileName, size_t fSize); 

/* Memory allocation and processing */
void PrepareArrays();
void CreateFloatData(); 	
void WrapAroundVolume(); 
void CreateSpectrum(); 
void WrapAroundSpectrum(); 	
void CreateSpectrumTexture(); 
void GetSpectrumSlice(); 
void PackingSpectrumTexture(); 
void SendSpectrumTextureToGPU(); 
void SetDisplayList(); 	

/* Wrapping-around & repacking on CPU */
float**  FFT_Shift_2D(float** inputArr, float** outputArr, int N);
float*** FFT_Shift_3D(float* inputArray, int N);
float* 	Repack_2D(float** Input_2D, float* Input_1D, int N);
float* 	Repack_3D(float*** Input_3D, float* Input_1D, int N);

/* CUDA slice-rocessing "high-order reconstruction" */ 
extern "C"
void LaunchProcessingCUDA (dim3 fGrid, dim3 fBlock, cudaArray* fInputSlice, float* fOutputImage, int fImgWidth, int fImgHeight, cufftComplex* fInputComplexData, cufftComplex *fOutputComplexData, float* TempArray, int ScaleFactor);

/* CUDA renderer */
extern "C"
void CUDA_Way(); 

/* CUDA volume processing */
extern "C" 
void LaunchVolumeProcessingCUDA(cufftComplex* sDeviceComplexArray, cufftComplex* sDeviceComplexArrayTemp, int sDataSize, float* Dev1DArray); 

/* CUDA context functions */ 
void ProcessSlice();

/* Context cleanning */ 
void CleanUp(int fExitCode);

/* Initializaion of CUDA/GL functionality */ 
bool InitCUDAContext(int argc, char **argv, bool fUseOpenGL);
bool InitOpenGLContext(int *argc, char** argv);
void InitOpenGLBuffers();

/* Creating / deleting OpenGL pixel buffer objects */ 
void CreatePBO(GLuint* fPBO, struct cudaGraphicsResource **fPBO_Resource);
void DeletePBO(GLuint* fPBO);

/* Creating / deleting OpenGL textures */ 
void CreatetextureDst(GLuint* fTextureID, unsigned int xSize, unsigned int ySize);
void CreateTextureSrc(GLuint* fTextureID, unsigned int xSize, unsigned int ySize);
void DeleteTexture(GLuint* fTex);

/* Creating / deleting OpenGL framebuffer object */ 
void CreateFrameBuffer(GLuint* fFBO, GLuint fColor);
void DeleteFrameBuffer(GLuint* fFBO);

/* OpenGL callBacks */ 
void DisplayGL();
void IdleGL();
void KeyBoardGL(unsigned char fKey, int x, int y);
void ReshapeGL(int fWisth, int fHeight);
void MainMenuGL(int iSelection);

#endif // _FOURIER_VOLUME_RENDERER_H
