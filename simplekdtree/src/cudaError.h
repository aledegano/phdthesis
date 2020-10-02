/*
 * cudaError.h
 *
 *  Created on: Jun 13, 2013
 *      Author: Felice Pantaleo
 */

#include <stdio.h>

#ifndef CUDAERROR_H_
#define CUDAERROR_H_

// Define this to turn on error checking

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
//#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
//#endif

return;
}

inline void __cudaCheckError( const char *file, const int line )
{
//#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}

// More careful checking. However, this will affect performance.
// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
//#endif

return;
}


#endif /* CUDAERROR_H_ */
