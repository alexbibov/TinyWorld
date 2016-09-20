#include "CudaErrorWrapper.h"


using namespace CudaUtils;


CudaErrorWrapper::CudaErrorWrapper() : error_code{ cudaSuccess }, error_file("unknown"), error_func("unknown()"), error_line{ 0xFFFFFFFF }
{

}


CudaErrorWrapper::CudaErrorWrapper(cudaError_t error_code, std::string error_file, std::string error_func, uint32_t error_line) :
error_code{ error_code }, error_file(error_file), error_func(error_func), error_line{ error_line }
{

}


CudaErrorWrapper::operator bool() const
{
	return error_code != cudaSuccess;
}


std::string CudaErrorWrapper::to_string() const
{
	return std::string{ "CUDA error in \"" } +error_file + "\", in function \"" + error_func + "(...)\", line " + std::to_string(error_line) + ": " +
		cudaGetErrorString(error_code);
}


const char* CudaErrorWrapper::getCudaErrorDescription() const
{
	return cudaGetErrorString(error_code);
}


const char* CudaErrorWrapper::getErrorFile() const
{
	return error_file.c_str();
}


const char* CudaErrorWrapper::getErrorFunction() const
{
	return error_func.c_str();
}


uint32_t CudaErrorWrapper::getErrorLine() const
{
	return error_line;
}