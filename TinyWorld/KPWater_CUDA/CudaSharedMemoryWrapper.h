#ifndef CUDA_SHARED_MEMORY_WRAPPER_INCLUDED

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


//This template is a workaround to deal with the fact that shared memory blocks are
//in fact defined as global by current nVidia tool-chains (up to CUDA 7.0). This prevents
//usage of template types on shared memory block declarations
template<typename T> class CudaSharedMemoryWrapper
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static T* getSharedMemoryBlockPointer()
	{
		static_assert(true, "undefined specialization of CUDASharedMemoryWrapper__");
		return nullptr;
	}
};

template<>
class CudaSharedMemoryWrapper < float > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static float* getSharedMemoryBlockPointer()
	{
		extern __shared__ float float_mem_block[];
		return float_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < double > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static double* getSharedMemoryBlockPointer()
	{
		extern __shared__ double double_mem_block[];
		return double_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < long double > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static long double* getSharedMemoryBlockPointer()
	{
		extern __shared__ long double long_double_mem_block[];
		return long_double_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < bool > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static bool* getSharedMemoryBlockPointer()
	{
		extern __shared__ bool bool_mem_block[];
		return bool_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < char > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static char* getSharedMemoryBlockPointer()
	{
		extern __shared__ char char_mem_block[];
		return char_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < unsigned char > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static unsigned char* getSharedMemoryBlockPointer()
	{
		extern __shared__ unsigned char uchar_mem_block[];
		return uchar_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < wchar_t > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static wchar_t* getSharedMemoryBlockPointer()
	{
		extern __shared__ wchar_t wchar_t_mem_block[];
		return wchar_t_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < short int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static short int* getSharedMemoryBlockPointer()
	{
		extern __shared__ short int shortint_mem_block[];
		return shortint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < unsigned short int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static unsigned short int* getSharedMemoryBlockPointer()
	{
		extern __shared__ unsigned short int ushortint_mem_block[];
		return ushortint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static int* getSharedMemoryBlockPointer()
	{
		extern __shared__ int int_mem_block[];
		return int_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < unsigned int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static unsigned int* getSharedMemoryBlockPointer()
	{
		extern __shared__ unsigned int uint_mem_block[];
		return uint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < long int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static long int* getSharedMemoryBlockPointer()
	{
		extern __shared__ long int longint_mem_block[];
		return longint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < unsigned long int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static unsigned long int* getSharedMemoryBlockPointer()
	{
		extern __shared__ unsigned long int ulongint_mem_block[];
		return ulongint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < long long int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static long long int* getSharedMemoryBlockPointer()
	{
		extern __shared__ long long int longlongint_mem_block[];
		return longlongint_mem_block;
	}
};

template<>
class CudaSharedMemoryWrapper < unsigned long long int > final
{
private:
	CudaSharedMemoryWrapper() {};
	CudaSharedMemoryWrapper(const CudaSharedMemoryWrapper& other) = delete;
	CudaSharedMemoryWrapper& operator=(const CudaSharedMemoryWrapper& other) = delete;

public:
	__device__ static unsigned long long int* getSharedMemoryBlockPointer()
	{
		extern __shared__ unsigned long long int ulonglongint_mem_block[];
		return ulonglongint_mem_block;
	}
};


#define CUDA_SHARED_MEMORY_WRAPPER_INCLUDED
#endif