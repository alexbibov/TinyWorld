#include <cuda_runtime.h>
#include <cstdlib>
#include <type_traits>
#include <exception>

#include "UnifiedMemoryBlock.h"
#include "CudaDeviceScheduler.h"

using namespace CudaUtils;


UnifiedMemoryBlock::UnifiedMemoryBlock() : block_size{ 0 }, allocation_carret{ 0 }, block_start_addr{ nullptr }
{
	
}


UnifiedMemoryBlock::UnifiedMemoryBlock(size_t size) : block_size{ size }, allocation_carret{ 0 }, block_start_addr{ nullptr }
{
	
}


UnifiedMemoryBlock::UnifiedMemoryBlock(const UnifiedMemoryBlock& other) :
block_size{ other.block_size }, allocation_carret{ other.allocation_carret }, block_start_addr{ nullptr }
{
	
}


UnifiedMemoryBlock::UnifiedMemoryBlock(UnifiedMemoryBlock&& other) :
block_size{ other.block_size }, allocation_carret{ other.allocation_carret }, block_start_addr{ other.block_start_addr }
{
	other.block_start_addr = nullptr;
}


UnifiedMemoryBlock& UnifiedMemoryBlock::operator =(const UnifiedMemoryBlock& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	block_size = other.block_size;
	allocation_carret = other.allocation_carret;

	return *this;
}


UnifiedMemoryBlock& UnifiedMemoryBlock::operator =(UnifiedMemoryBlock&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	block_size = other.block_size;
	allocation_carret = other.allocation_carret;
	std::swap(block_start_addr, other.block_start_addr);

	return *this;
}


UnifiedMemoryBlock::~UnifiedMemoryBlock()
{
	
}


void UnifiedMemoryBlock::initialize(size_t size)
{
	if (!block_start_addr)
	{
		if (size == 0)
			return;

		block_start_addr = perform_allocation(size);
		block_size = size;
	}
}


void* UnifiedMemoryBlock::allocate(size_t size)
{
	//Allocation of memory chunks with zero sizes are not allowed
	if (size == 0)
		return nullptr;

	//If there is not enough space left in the buffer to perform allocation return invalid pointer
	if (size > block_size - allocation_carret)
		return nullptr;

	//Compute starting address of the new allocation
	void* rv = static_cast<char*>(block_start_addr)+allocation_carret;
	allocation_carret += size;	//move allocation caret forward

	return rv;
}


void* UnifiedMemoryBlock::navigate(size_t offset) const
{
	//If the block has not been initialized return invalid pointer
	if (!block_start_addr)
		return nullptr;

	//If offset is out of the allowed natural range return invalid pointer
	if (offset >= block_size)
		return nullptr;

	return static_cast<char*>(block_start_addr) + offset;
}


void UnifiedMemoryBlock::reset()
{
	allocation_carret = 0;
}


void UnifiedMemoryBlock::release()
{
	if (perform_deallocation())
	{
		block_size = 0;
		allocation_carret = 0;
		block_start_addr = nullptr;
	}
}


size_t UnifiedMemoryBlock::getSize() const
{
	return isInitialized() ? block_size : 0;
}


bool UnifiedMemoryBlock::isInitialized() const
{
	return block_start_addr != nullptr;
}




void* CPUMemoryBlock::perform_allocation(size_t allocation_size)
{
	void* rv = malloc(allocation_size);
	return rv ? rv : nullptr;
}


bool CPUMemoryBlock::perform_deallocation()
{
	if (block_start_addr)
	{
		free(block_start_addr);
		return true;
	}

	return false;
}


CPUMemoryBlock::CPUMemoryBlock() : UnifiedMemoryBlock()
{
	
}


CPUMemoryBlock::CPUMemoryBlock(size_t size) : UnifiedMemoryBlock(size)
{
	block_start_addr = perform_allocation(size);
}


CPUMemoryBlock::CPUMemoryBlock(const CPUMemoryBlock& other) :
UnifiedMemoryBlock(other)
{
	if (other.isInitialized() && (block_start_addr = perform_allocation(other.getSize())))
		memcpy(block_start_addr, other.block_start_addr, other.getSize());
}


CPUMemoryBlock::CPUMemoryBlock(CPUMemoryBlock&& other) :
UnifiedMemoryBlock(std::move(other))
{
	
}


CPUMemoryBlock& CPUMemoryBlock::operator=(const CPUMemoryBlock& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//If the "other" object is not initialized, release the allocation owned by "this" object so it
	//gets reset to its default state
	if (!other.isInitialized())
	{
		release();
		return *this;
	}

	//If "this" object already owns a memory allocation large enough to hold the data from the"other" object,
	//just copy the data from the "other" object into "this" object. Otherwise, ensure that the receiving object
	//possesses sufficient allocation space
	if (getSize() < other.getSize())
	{
		perform_deallocation();
		if (!(block_start_addr = perform_allocation(other.getSize())))
			return *this;
	}
	memcpy(block_start_addr, other.block_start_addr, other.getSize());

	UnifiedMemoryBlock::operator=(other);

	return *this;
}


CPUMemoryBlock& CPUMemoryBlock::operator=(CPUMemoryBlock&& other)
{
	if (this == &other)
		return *this;

	UnifiedMemoryBlock::operator=(std::move(other));
	return *this;
}


CPUMemoryBlock::~CPUMemoryBlock()
{
	perform_deallocation();
}


CPUMemoryBlock::location CPUMemoryBlock::getLocation() const
{
	return location::CPU;
}




void* GPUMemoryBlock::perform_allocation(size_t allocation_size)
{
	void* p_buf;
	CudaErrorWrapper aux_err_desc;
	if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(gpu_id))) ||
		(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaMalloc(&p_buf, allocation_size))))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = gpu_id;
		error_callback(error_descriptor, error_source);
		return nullptr;
	}

	return p_buf;
}


bool GPUMemoryBlock::perform_deallocation()
{
	if (!block_start_addr) return false;

	CudaErrorWrapper aux_err_desc;
	if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(gpu_id))) || 
		(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaFree(block_start_addr))))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = gpu_id;
		error_callback(error_descriptor, error_source);
		return false;
	}

	return true;
}


GPUMemoryBlock::GPUMemoryBlock() : UnifiedMemoryBlock(), gpu_id{ -1 },
error_state{ false }, error_callback{ [](const CudaErrorWrapper& err, int dev_id)->void{} }
{
	
}


GPUMemoryBlock::GPUMemoryBlock(int gpu_id, size_t size) : UnifiedMemoryBlock(size), gpu_id{ gpu_id },
error_state{ false }, error_callback{ [](const CudaErrorWrapper& err, int dev_id)->void{} }
{
	CudaErrorWrapper aux_err_desc;

	int current_device;
	if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaGetDevice(&current_device))) 
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = -1;
		error_callback(error_descriptor, error_source);
		return;
	}

	block_start_addr = perform_allocation(size);

	if (block_start_addr && 
		(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(current_device))))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = current_device;
		error_callback(error_descriptor, error_source);

		if (!perform_deallocation())
			throw(std::runtime_error{ "Unable to initialize GPU memory buffer" });

		block_start_addr = nullptr;
	}
}


GPUMemoryBlock::GPUMemoryBlock(const GPUMemoryBlock& other) :
UnifiedMemoryBlock(other), gpu_id{ other.gpu_id }, 
error_state{ other.error_state }, error_descriptor{ other.error_descriptor }, error_callback{ other.error_callback }
{
	if (other.isInitialized())
	{
		CudaErrorWrapper aux_err_desc;

		int current_device;
		if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaGetDevice(&current_device)))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = -1;
			error_callback(error_descriptor, error_source);
			return;
		}


		block_start_addr = perform_allocation(other.getSize());


		if (block_start_addr &&
			(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaMemcpy(block_start_addr, other.block_start_addr, 
			other.getSize(), cudaMemcpyDeviceToDevice))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = gpu_id;
			error_callback(error_descriptor, error_source);

			if (!perform_deallocation())
				throw(std::runtime_error{ "Unable to copy GPU data" });

			if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(current_device)))
			{
				error_descriptor = aux_err_desc;
				error_source = current_device;
				error_callback(error_descriptor, error_source);
			}

			block_start_addr = nullptr;
			return;
		}


		if (block_start_addr &&
			(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(current_device))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = current_device;
			error_callback(error_descriptor, error_source);

			if (!perform_deallocation())
				throw(std::runtime_error{ "Unable to copy GPU data" });

			block_start_addr = nullptr;
			return;
		}
	}
}


GPUMemoryBlock::GPUMemoryBlock(GPUMemoryBlock&& other) :
UnifiedMemoryBlock(std::move(other)), gpu_id{ other.gpu_id }, 
error_state{ other.error_state }, error_descriptor{ std::move(other.error_descriptor) },
error_callback{ std::move(other.error_callback) }
{

}


GPUMemoryBlock& GPUMemoryBlock::operator=(const GPUMemoryBlock& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//If the "other" object was not initialized, ensure that "this" object is reset to its default state
	if (!other.isInitialized())
	{
		release();
		return *this;
	}


	CudaErrorWrapper aux_err_desc;

	//Retrieve currently active device. In case of an error, put object into "uninitialized" state
	int current_device;
	if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaGetDevice(&current_device)))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = -1;
		error_callback(error_descriptor, error_source);

		if (block_start_addr && !perform_deallocation())
			throw(std::runtime_error{ "Unable to perform GPU block copy-assignment operation" });


		block_start_addr = nullptr;
		return *this;
	}


	//If the "other" object is initialized, perform the copy-assignment, which is dependent on the size of "this" and
	//the "other" blocks and on whether the blocks are allocated on the same GPU
	if (gpu_id != other.gpu_id || getSize() < other.getSize())
	{
		if (isInitialized())
		{
			if (!perform_deallocation())
				throw(std::runtime_error{ "Unable to perform GPU block copy-assignment operation" });
			block_start_addr = nullptr;
		}

		gpu_id = other.gpu_id;
		if (!(block_start_addr = perform_allocation(other.getSize())))
			return *this;
	}
	else
	{
		if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(gpu_id)))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = gpu_id;
			error_callback(error_descriptor, error_source);

			if (!perform_deallocation())
				throw(std::runtime_error{ "Unable to perform GPU block copy-assignment operation" });
			block_start_addr = nullptr;
			return *this;
		}
	}


	//Copy data from the "other" object to "this" object
	if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaMemcpy(block_start_addr, other.block_start_addr, other.getSize(), cudaMemcpyDeviceToDevice)))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = gpu_id;
		error_callback(error_descriptor, error_source);

		if (!perform_deallocation())
			throw(std::runtime_error{ "Unable to perform GPU copy-assignment operation" });

		block_start_addr = nullptr;

		return *this;
	}

	//Restore CUDA context
	if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(current_device)))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = current_device;
		error_callback(error_descriptor, error_source);

		if (!perform_deallocation())
			throw(std::runtime_error{ "Unable to perform GPU copy-assignment operation" });

		block_start_addr = nullptr;

		return *this;
	}

	UnifiedMemoryBlock::operator=(other);

	return *this;
}


GPUMemoryBlock& GPUMemoryBlock::operator=(GPUMemoryBlock&& other)
{
	if (this == &other)
		return *this;

	UnifiedMemoryBlock::operator=(std::move(other));
	std::swap(gpu_id, other.gpu_id);
	error_state = other.error_state;
	error_descriptor = std::move(other.error_descriptor);
	error_source = other.error_source;
	error_callback = std::move(other.error_callback);
	return *this;
}


GPUMemoryBlock::~GPUMemoryBlock()
{
	if (block_start_addr)
	{
		//Here we do not need to change the error state of the object as it gets destroyed anyway
		//But it still can send the final "good bye!" via the error callback
		CudaErrorWrapper aux_err_desc;
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(gpu_id))) &&
			(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaFree(block_start_addr))))
			error_callback(aux_err_desc, gpu_id);
	}
}


GPUMemoryBlock::operator bool() const
{
	return !error_state;
}


void GPUMemoryBlock::initialize(size_t size)
{
	initialize(CudaDeviceScheduler{}.bestDevice()->getCudaDeviceId(), size);
}


void GPUMemoryBlock::initialize(int gpu_id, size_t size)
{
	this->gpu_id = gpu_id;

	CudaErrorWrapper aux_err_desc;

	//Retrieve numerical identifier of the device, which is currently active on the given CUDA context
	int current_device;
	if (aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaGetDevice(&current_device)))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = -1;
		error_callback(error_descriptor, error_source);
		return;
	}

	//Allocate memory for the block
	UnifiedMemoryBlock::initialize(size);

	if (block_start_addr &&
		(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(current_device))))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = current_device;
		error_callback(error_descriptor, error_source);

		if (!perform_deallocation())
			throw(std::runtime_error{ "Unable to allocate buffer for GPU memory block object" });

		block_start_addr = nullptr;
		return;
	}
}


GPUMemoryBlock::location GPUMemoryBlock::getLocation() const
{
	return location::GPU;
}


int GPUMemoryBlock::getOwningDeviceId() const
{ 
	return isInitialized() ? gpu_id : -1;
}


void GPUMemoryBlock::registerErrorCallback(const std::function<void(const CudaErrorWrapper&, int)>& error_callback)
{
	this->error_callback = error_callback;
}


std::pair<CudaErrorWrapper, int> GPUMemoryBlock::getLastError() const
{
	return std::make_pair(error_descriptor, error_source);
}


bool GPUMemoryBlock::getErrorState() const { return error_state; }



















