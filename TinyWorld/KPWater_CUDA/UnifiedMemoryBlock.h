#ifndef CUDA_UTILS__UNIFIED_MEMORY_BLOCK__
#define CUDA_UTILS__UNIFIED_MEMORY_BLOCK__

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <cstdint>
#include <functional>

#include "CudaErrorWrapper.h"


namespace CudaUtils
{
	//Class that represents linear memory allocation model used by SVSCore
	class UnifiedMemoryBlock
	{
	private:
		size_t block_size;	//full size of the memory block
		size_t allocation_carret;	//offset, at which the next allocation will be performed

		//Performs actual allocation of a data chunk to be used by the block. Returns the
		//start address of the allocation. In case of failure returns nullptr
		virtual void* perform_allocation(size_t allocation_size) = 0;

		//Deallocates memory chunk owned by the block. Returns 'true' on success and 'false' on failure
		virtual bool perform_deallocation() = 0;

	protected:
		void* block_start_addr;	//start address of the block


		//Initializes new memory block with no data allocated
		UnifiedMemoryBlock();

		//Initializes new memory block of the given size
		explicit UnifiedMemoryBlock(size_t size);

		//Copy constructor
		UnifiedMemoryBlock(const UnifiedMemoryBlock& other);

		//Move constructor
		UnifiedMemoryBlock(UnifiedMemoryBlock&& other);

		//Copy assignment operator
		UnifiedMemoryBlock& operator=(const UnifiedMemoryBlock& other);

		//Move assignment operator
		UnifiedMemoryBlock& operator=(UnifiedMemoryBlock&& other);

	public:
		enum class location{ CPU, GPU };	//describes physical location of the data in the block


		//Destructor
		virtual ~UnifiedMemoryBlock();

		//Initializes the memory block and allocates for it a memory chunk with given size.
		//If the block has already been initialized, repetitive initialization will have no effect.
		void initialize(size_t size);

		//Allocates requested amount of data from the memory block. If allocation can not be done
		//returns nullptr
		void* allocate(size_t size);

		//Returns address in memory that corresponds to the given offset from the beginning of the block.
		//If such address can not be computed (for instance, when requested offset is greater than the size of the block)
		//the function returns nullptr. Note that this function does not change position of the allocation caret of the block
		void* navigate(size_t offset) const;

		//Resets the allocation caret of the memory block. If the allocation caret has been reset, subsequent
		//allocations will start exploiting the block from the zero-offset (effectively overwriting the data contained in it)
		void reset();

		//Releases all data contained in the block. After calling this function the block will be having 
		//"not initialized" status and could not be used before calling initialize(...)
		void release();

		//Returns physical location of the data held by the block
		virtual location getLocation() const = 0;

		//Returns size of the block represented in bytes. If block has not been initialized returns 0
		size_t getSize() const;

		//Returns 'true' if the block has been initialized, returns 'false' otherwise
		bool isInitialized() const;
	};


	//Class representing allocation hosted on the CPU-side
	class CPUMemoryBlock final : public UnifiedMemoryBlock
	{
	private:
		//Performs actual allocation of a data chunk to be used by the block. Returns the
		//start address of the allocation. In case of failure returns nullptr
		void* perform_allocation(size_t allocation_size) override;

		//Deallocates memory chunk owned by the block. Returns 'true' on success and 'false' on failure
		bool perform_deallocation() override;

	public:
		//Initializes new memory block with no data allocated
		CPUMemoryBlock();

		//Initializes new memory block of the given size
		explicit CPUMemoryBlock(size_t size);

		//Copy constructor
		CPUMemoryBlock(const CPUMemoryBlock& other);

		//Move constructor
		CPUMemoryBlock(CPUMemoryBlock&& other);

		//Copy assignment operator
		CPUMemoryBlock& operator=(const CPUMemoryBlock& other);

		//Move assignment operator
		CPUMemoryBlock& operator=(CPUMemoryBlock&& other);

		//Destructor
		~CPUMemoryBlock();

		//Returns physical location of the data held by the block
		location getLocation() const override;
	};


	//Class representing allocation hosted on the GPU-side
	class GPUMemoryBlock final : public UnifiedMemoryBlock
	{
	private:
		int gpu_id;	//identifier of the device owning the data
		bool error_state;	//equals 'true' if object resides in an erroneous state, equals 'false' otherwise
		CudaErrorWrapper error_descriptor;		//objects describing the last error occurred when using the GPU memory block
		int error_source;	//numerical identifier of the CUDA-capable device that has been the source of the last error
		std::function<void(const CudaErrorWrapper&, int)> error_callback;	//callback function which is called in case of an error

		//Performs actual allocation of a data chunk to be used by the block. Returns the
		//start address of the allocation. In case of failure returns nullptr. Note that the function is not context-safe meaning that
		//it will automatically set current device to "gpu_id" and will not restore the configuration that has been active prior to the function call.
		void* perform_allocation(size_t allocation_size) override;

		//Deallocates memory chunk owned by the block. Returns 'true' on success and 'false' on failure. Note that the function is not 
		//context-safe meaning that it will automatically set current device to "gpu_id" and will not restore the configuration that has been 
		//active prior to the function call.
		bool perform_deallocation() override;

	public:
		//Initializes new memory block with no data allocated
		GPUMemoryBlock();

		//Initializes new memory block of the given size allocated on GPU with the given numerical identifier
		GPUMemoryBlock(int gpu_id, size_t size);

		//Copy constructor
		GPUMemoryBlock(const GPUMemoryBlock& other);

		//Move constructor
		GPUMemoryBlock(GPUMemoryBlock&& other);

		//Copy assignment operator
		GPUMemoryBlock& operator=(const GPUMemoryBlock& other);

		//Move assignment operator
		GPUMemoryBlock& operator=(GPUMemoryBlock&& other);

		//Destructor
		~GPUMemoryBlock();

		//Returns 'true' if the object is NOT in an erroneous state. Returns 'false' otherwise
		operator bool() const;

		//Initializes the memory block and allocates for it a memory chunk with given size.
		//If the block has already been initialized, repetitive initialization will have no effect.
		//This function allocates memory from the "best" CUDA-capable device present on the host system.
		void initialize(size_t size);

		//Initializes memory block and allocates a memory segment of the given size of its uses.
		//The memory segment is allocated from device with the given numerical identifier.
		//If the memory block has already been initializes the function has no effect.
		void initialize(int gpu_id, size_t size);

		//Returns physical location of the data held by the block
		location getLocation() const override;

		//Returns numerical identifier of the GPU owning the data. If the owning GPU can not be determined (for
		//example in case when the block has not yet been initialized) the function returns -1.
		int getOwningDeviceId() const;

		//Registers callback function, which will be called on error occurrences. When the object runs into an error
		//it firstly calls the error callback and when the execution flow if returned back to the object it tries to
		//revert it back to uninitialized state by releasing its memory allocation. If even this is impossible due to some
		//severe CUDA context damage, the object throws a runtime exception
		void registerErrorCallback(const std::function<void(const CudaErrorWrapper&, int)>& error_callback);

		//Returns pair with the first element containing error descriptor object of the last CUDA-related error, and the second
		//elements equal to numerical identifier of the CUDA-capable device that has been the source of the last error
		std::pair<CudaErrorWrapper, int> getLastError() const;

		//Returns error state of the object
		bool getErrorState() const;
	};
}



#endif 


