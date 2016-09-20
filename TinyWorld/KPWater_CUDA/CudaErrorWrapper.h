#ifndef CUDA_UTILS__CUDA_ERROR_WRAPPER__
#define CUDA_UTILS__CUDA_ERROR_WRAPPER__


#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string>
#include <cstdint>


namespace CudaUtils
{
	//Represents CUDA-related error
	class CudaErrorWrapper final
	{
	private:
		cudaError_t error_code;
		std::string error_file;
		std::string error_func;
		uint32_t error_line;

	public:
		CudaErrorWrapper();
		CudaErrorWrapper(cudaError_t error_code, std::string error_file, std::string error_func, uint32_t error_line);

		operator bool() const;
		std::string to_string() const;

		const char* getCudaErrorDescription() const;
		const char* getErrorFile() const;
		const char* getErrorFunction() const;
		uint32_t getErrorLine() const;
	};


	inline std::ostream& operator<<(std::ostream& output_stream, CudaErrorWrapper& cuda_error_object) { return output_stream << cuda_error_object.to_string(); }


#ifndef __func__ 
#define __func__ __FUNCTION__
#endif

	//Safety wrapper for CUDA-related function calls
	inline CudaErrorWrapper cuda_safe_call__(cudaError_t cuda_error_code, std::string file, std::string func, uint32_t line)
	{
		return cuda_error_code == cudaSuccess ? CudaErrorWrapper{} : CudaErrorWrapper{ cuda_error_code, file, func, line };
	}

#define CUDA_SAFE_CALL(cuda_error_code)\
			{CudaError __cuda_safe_call_macro_err_obj; if(__cuda_safe_call_macro_err_obj = cuda_safe_call__((cuda_error_code), __FILE__, __func__, __LINE__)) return __cuda_safe_call_macro_err_obj;}

#define CUDA_SAFE_CALL_NO_RETURN(cuda_error_code)cuda_safe_call__((cuda_error_code), __FILE__, __func__, __LINE__)
}


#endif