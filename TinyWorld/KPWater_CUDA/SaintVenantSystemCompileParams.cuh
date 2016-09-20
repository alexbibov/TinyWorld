#ifndef SVS__COMPILE_PARAMS__
#define SVS__COMPILE_PARAMS__


//Precision of the computations. Comment this line and uncomment "#define PRECISION_DOUBLE" 
//to compile in double-precision mode.
#define SVS_PRECISION_FLOAT
//#define SVS_PRECISION_DOUBLE


//Dimension of a single CUDA computational block. The default value is 16-by-16.
//You can try to modify this parameter in order to achieve better performance on your device.
#define SVS_CUDA_BLOCK_SIZE 16

//Binary logarithm of SVS_CUDA_BLOCK_SIZE
#define SVS_LOG_CUDA_BLOCK_SIZE 4

#endif
