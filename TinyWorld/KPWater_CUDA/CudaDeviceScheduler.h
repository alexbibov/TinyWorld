#ifndef CUDA_UTILS__CUDA_DEVICE_SCHEDULER__
#define CUDA_UTILS__CUDA_DEVICE_SCHEDULER__

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>


namespace CudaUtils
{
	//Describes particulars of a CUDA-capable device
	class CudaDeviceTraits
	{
	private:
		uint32_t cuda_id;	//CUDA identifier of the device
		char name[256];		//device's string name
		size_t totalGlobalMem;		//total amount of global memory provided by device
		size_t sharedMemPerBlock;	//amount of shared memory available for each block
		int regsPerBlock;	//amount of 32-bit registers available per block
		int warpSize;		//size of a single warp in threads
		size_t memPitch;	//maximum pitch in bytes allowed by memory copies
		int maxThreadsPerBlock;	//maximum number of threads that can be wrapped up in a single block
		int maxThreadsDim[3];	//maximum number of threads per each dimension of a block
		int maxGridSize[3];	//maximum size of each dimension of a grid
		int clockRate;		//GPU clock rate represented in kilohertz
		size_t totalConstMem;	//constant memory available on device in bytes
		int major;		//major compute capability
		int minor;		//minor compute capability
		size_t textureAlignment;	//alignment requirement for textures
		size_t texturePitchAlignment;	//pitch alignment requirement for texture references bound to pitched memory
		int multiProcessorCount;	//number of multiprocessors on a device
		int kernelExecTimeoutEnabled;	//specifies whether there is a run time limit set for kernel execution
		int integrated;	//device is integrated (as opposed to discrete)
		int canMapHostMemory_;	//device is able to map host memory using cudaHostAlloc/cudaHostGetDevicePointer
		int computeMode;	//compute mode used by the device (supported by Tesla series)
		int maxTexture1D;	//maximum 1D texture size
		int maxTexture1DMipmap;	//maximum 1D mipmapped texture size
		int maxTexture1DLinear;	//maximum size for 1D textures bound to linear memory
		int maxTexture2D[2];	//maximum 2D texture dimensions
		int maxTexture2DMipmap[2];	//maximum 2D mipmapped texture dimensions
		int maxTexture2DLinear[3];	//maximum dimensions (width, height, and pitch) for 2D textures bound to pitched memory
		int maxTexture2DGather[2];	//maximum 2D texture dimensions if texture gather operations have to be performed 
		int maxTexture3D[3];	//maximum 3D texture dimensions
		int maxTexture3DAlt[3];	//maximum alternate 3D texture dimensions
		int maxTextureCubemap;	//maximum cubemap texture dimensions
		int maxTexture1DLayered[2];	//maximum 1D layered texture dimensions
		int maxTexture2DLayered[3];	//maximum 2D layered texture dimensions
		int maxTextureCubemapLayered[2];	//maximum cubemap layered texture dimensions
		int maxSurface1D;	//maximum 1D surface size
		int maxSurface2D[2];	//maximum 2D surface dimensions
		int maxSurface3D[3];	//maximum 3D surface dimensions
		int maxSurface1DLayered[2];	//maximum 1D layered surface dimensions;
		int maxSurface2DLayered[3];	//maximum 2D layered surface dimensions;
		int maxSurfaceCubemap;	//maximum cubemap surface dimensions
		int maxSurfaceCubemapLayered[2];	//maximum cubemap layered surface dimensions
		size_t surfaceAlignment;	//alignment requirements for surfaces
		int concurrentKernels;	//device can possibly execute multiple kernels concurrently
		int ECCEnabled;	//device has ECC support enabled
		int pciBusID;	//PCI bus ID of the device
		int pciDeviceID;	//PCI device ID of the device
		int pciDomainID;	//PCI domain ID of the device
		int tccDriver;		//1 if device is a Tesla device using TCC driver, 0 otherwise
		int asyncEngineCount;	//number of asynchronous engines
		int unifiedAddressing;	//device shares a unified address space with the host
		int memoryClockRate;	//peak memory clock frequency in kilohertz
		int memoryBusWidth;	//global memory bus width in bits
		int l2CacheSize;		//size of L2 cache in bytes
		int maxThreadsPerMultiProcessor;	//maximum resident threads per multiprocessor
		int streamPrioritiesSupported;	//device supports stream priorities
		int globalL1CacheSupported;	//device supports caching globals in L1
		int localL1CacheSupported;		//device supports caching locals in L1
		size_t sharedMemPerMultiProcessor;	//shared memory available per multiprocessor in bytes
		int regsPerMultiProcessor;		//32-bit registers available per multiprocessor
		int isMultiGpuBoard;	//device is on a multi-GPU board
		int multiGpuBoardGroupID;	//unique identifier for a group of devices on the same multi-GPU board

		bool error_state;		//equals 'true' if object is in erroneous state. Equals 'false' otherwise
		std::string error_string;	//contains description of the last occurred error

	public:
		explicit CudaDeviceTraits(uint32_t cuda_device_id);		//initializes device traits using given CUDA id of the device

		//Returns CUDA identifier of the device
		uint32_t getCudaDeviceId() const;

		//Returns string name of the device
		std::string getDeviceName() const;

		//Returns total on-board memory installed on the device
		size_t getTotalGlobalMemory() const;

		//Returns amount of shared memory available per block for the given device
		size_t getSharedMemoryPerBlock() const;

		//Returns amount of 32-bit registers available per block for the device
		int getNumberOf32BitRegistersPerBlock() const;

		//Returns size of a CUDA-warp for the device
		int getWarpSize() const;

		//Returns maximum pitch represented in bytes allowed by memory copies
		size_t getMemoryPitchSize() const;

		//Returns maximum number of threads that can be run within a single block
		int getNumberOfThreadsPerBlock() const;

		//Returns maximum number of threads per each dimension of a block
		dim3 getNumberOfThreadsPerBlockDimension() const;

		//Returns maximum number of blocks per each dimension of a CUDA grid
		dim3 getNumberOfBlocksInGrid() const;

		//Returns clock rate of the GPU in kilohertz
		uint32_t getGpuClockRate() const;

		//Returns amount of available constant memory represented in bytes
		size_t getConstantMemory() const;

		//Returns major and minor parts of compute capability of the device
		std::pair<uint32_t, uint32_t> getComputeCapability() const;

		//Returns alignment requirement for textures
		size_t getTextureAlignment() const;

		//Returns pitch alignment requirements for texture references bound to pitch memory
		size_t getTexturePitchAlignment() const;

		//Returns number of multiprocessors on the device
		uint32_t getNumberOfMultiprocessors() const;

		//Returns 'true' if kernel execution time out has been enabled on the host
		bool isKernelExecutionTimeoutEnabled() const;

		//Returns 'true' if the device is integrated (as opposed to discrete)
		bool isIntegrated() const;

		//Returns 'true' if device is able to map host memory using cudaHostAlloc/cudaHostGetDevicePointer
		bool canMapHostMemory() const;

		//Returns compute mode used by the device. This is only supported by Tesla-series and the
		//following compute modes are allowed:
		//cudaComputeModeDefault=0				Multiple threads can use cudaSetDevice() with this device
		//cudaComputeModeExclusive=1			Only one thread in one process will be able to use cudaSetDevice() with this device
		//cudaComputeModeProhibited=2			No threads can use cudaSetDevice() with this device
		//cudaComputeModeExclusiveProcess=3		Many threads in one process can will be able to use cudaSetDevice() with this device
		uint32_t getComputeMode() const;

		//Returns maximum size allowed for 1D-textures
		uint32_t getMaximum1DTextureSize() const;

		//Returns maximum size allowed for 1D mipmapped texture
		uint32_t getMaximum1DTextureMipmapSize() const;

		//Returns maximum size allowed for 1D textures bound to linear memory
		uint32_t getMaximum1DTextureLinearSize() const;

		//Returns maximum dimensions allowed for 2D texture (width and height are stored in 2-element array in this order)
		std::array<uint32_t, 2> getMaximum2DTextureSize() const;

		//Returns maximum dimensions allowed for 2D mipmapped texture (width and height are stored in 2-element array in this order)
		std::array<uint32_t, 2> getMaximum2DTextureMipmapSize() const;

		//Returns maximum dimensions allowed for 2D texture bound to pitch memory (returns 3-element array containing width, height, and pitch in this order)
		std::array<uint32_t, 3> getMaximum2DTextureLinearSize() const;

		//Returns maximum allowed dimensions of a 2D texture if texture gather operations have to be performed (width and height are stored in 2-element array in this order)
		std::array<uint32_t, 2> getMaximum2DTextureGatherSize() const;

		//Returns maximum allowed dimensions of a 3D texture (width, height and depth are stored to a 3-element array in this order)
		std::array<uint32_t, 3> getMaximum3DTextureSize() const;

		//Returns maximum allowed alternate 3D texture dimensions (width, height and depth are stored to a 3-element array in this order)
		std::array<uint32_t, 3> getMaximum3DTextureAlternateSize() const;

		//Returns maximum allowed cubemap texture dimensions
		uint32_t getMaximumCubemapTextureSize() const;

		//Returns maximum allowed 1D layered texture dimensions (size and number of layers are stored in this order in 2-element array returned by the function)
		std::array<uint32_t, 2> getMaximum1DLayeredTextureSize() const;

		//Returns maximum allowed 2D layered texture dimensions (width, height, and number of layers are stored in this order in 3-element array returned by the function)
		std::array<uint32_t, 3> getMaximum2DLayeredTextureSize() const;

		//Returns maximum allowed Cubemap layered texture dimensions
		std::array<uint32_t, 2> getMaximumCubemapLayeredTextureSize() const;

		//Returns maximum 1D surface size
		uint32_t getMaximum1DSurfaceSize() const;

		//Returns maximum 2D surface dimensions
		std::array<uint32_t, 2> getMaximum2DSurfaceSize() const;

		//Returns maximum 3D surface dimensions
		std::array<uint32_t, 3> getMaximum3DSurfaceSize() const;

		//Returns maximum 1D layered surface dimensions
		std::array<uint32_t, 2> getMaximum1DLayeredSurfaceSize() const;

		//Returns maximum 2D layered surface dimensions
		std::array<uint32_t, 3> getMaximum2DLayeredSurfaceSize() const;

		//Returns maximum cubemap surface dimensions
		uint32_t getMaximumSurfaceCubemapSize() const;

		//Returns maximum cubemap layered surface dimensions (the dimension and number of layers are packed into a 2-element array returned by the function)
		std::array<uint32_t, 2> getMaximumSurfaceCubemapLayeredSize() const;

		//Returns alignment requirement for surfaces
		size_t getSurfaceAlignmentRequirement() const;

		//Returns 'true' if device can possibly execute multiple kernels concurrently
		bool supportsConcurrentKernelExecution() const;

		//Returns 'true' if device has ECC support enabled
		bool isECCEnabled() const;

		//Returns PCI bus identifier of the device
		int getPCIBusId() const;

		//Returns PCI device identifier of the device
		int getPCIDeviceId() const;

		//Returns PCI domain identifier of the device
		int getPCIDomainId() const;

		//Returns 'true' if device belongs to Tesla series and runs a TCC-driver
		bool hasTCCDriver() const;

		//Returns number of asynchronous engines provided by the device
		uint32_t getNumberOfAsynchronousEngines() const;

		//Returns 'true' if device shares a unified address space with the host
		bool supportsUnifiedAddressSpace() const;

		//Returns peak memory clock rate in kilohertz
		uint32_t getPeakMemoryClockRate() const;

		//Returns global memory bus width represented in bits
		uint32_t getGlobalMemoryBusWidth() const;

		//Returns size of Level-2 cache memory in bytes
		size_t getLevel2CacheMemorySize() const;

		//Returns maximal resident threads per multi-processor
		size_t getMaximumThreadsPerMultiprocessor() const;

		//Returns 'true' if device supports stream priorities
		bool supportsStreamPriorities() const;

		//Returns 'true' if device supports caching globals in L1
		bool supportsL1GlobalCache() const;

		//Returns 'true' if device supports caching locals in L1
		bool supportsL1LocalCache() const;

		//Returns amount of shared memory available per multiprocessor represented in bytes
		size_t getSharedMemoryPerMultiprocessor() const;

		//Returns amount of 32-bit registers available per multiprocessor
		uint32_t get32BitRegistersPerMultiprocessor() const;

		//Returns 'true' if device is physically installed on a multi-GPU board
		bool isMultiGPU() const;

		//Returns unique identifier for a group of devices on the same multi-GPU board
		int getMultiGpuBoardIdentifier() const;

		operator bool() const;	//return 'true' if object is NOT in an erroneous state. Returns 'false' otherwise
		const char* getErrorString() const;		//returns string description of the last error
	};



	//The following flags allow to filter devices with desired properties
#define CDS_SUPPORTS_HOST_MEMORY_MAPPING			0x2			//enumerate devices that are able to map host memory using cudaHostAlloc/cudaHostGetDevicePointer
#define CDS_SUPPORTS_CONCURRENT_KERNEL_EXECUTION	0x4			//enumerate only those devices that support concurrent kernel execution
#define CDS_SUPPORTS_UNIFIED_ADDRESSING				0x20		//enumerate only those devices that support unified addressing
#define CDS_SUPPORTS_STREAM_PRIORITIES				0x40		//enumerate only those devices that support stream priorities
#define CDS_SUPPORTS_GLOBAL_L1_CACHE				0x80		//enumerate only those devices that support L1-caching from globals
#define CDS_SUPPORTS_LOCAL_L1_CACHE					0x100		//enumerate only those devices that support L1-caching from locals

#define CDS_DEVICE_DISCRETE			0x1		//enumerate only discrete devices
#define CDS_DEVICE_ECC_ENABLED		0x8		//enumerate only those devices that have ECC-enabled memory
#define CDS_DEVICE_TCC_DRIVER		0x10	//enumerate only Tesla devices
#define CDS_DEVICE_MULTI_GPU		0x200	//enumerate only those devices that are soldered on a multi-GPU boars



	class CudaDeviceScheduler
	{
	private:
		//Describes CUDA device iterator employed to enumerate all present CUDA-capable devices that meet given criteria.
		//This is a bidirectional iterator in accordance with the common terminology used in STL
		class CudaDeviceIterator : public std::iterator < std::forward_iterator_tag, CudaDeviceTraits >
		{
			friend class CudaDeviceScheduler;
		private:
			CudaDeviceScheduler* p_owner;		//pointer to the object, which handles the iterator
			int device_offset;		//offset to the target device traits

		public:
			explicit CudaDeviceIterator(CudaDeviceScheduler& owner);	//Default constructor
			CudaDeviceIterator(CudaDeviceScheduler& owner, int);	//Creates iterator that points to position after the last CUDA-capable device within the set of devices that meet proposed criteria. The second argument is a dummy integer to distinguish from the default constructor
			CudaDeviceIterator(const CudaDeviceIterator& other);	//Copy constructor

			//Provides access to the actual element
			CudaDeviceTraits operator*() const;

			//Provides access to a member of the actual element
			CudaDeviceTraits* operator->() const;

			//Steps forward and returns iterator pointing to the new position
			CudaDeviceIterator& operator++();

			//Steps forward and returns iterator pointing to the old position
			CudaDeviceIterator operator++(int);

			//Compares two iterators
			bool operator==(const CudaDeviceIterator& other) const;

			//Returns whether two iterators are not equal
			bool operator!=(const CudaDeviceIterator& other) const;

			//Assigns an iterator
			CudaDeviceIterator& operator=(const CudaDeviceIterator& other);

			//Steps backwards and returns iterator pointing to the new position
			CudaDeviceIterator& operator--();

			//Steps backwards and returns iterator pointing to the old position
			CudaDeviceIterator operator--(int);
		};



		uint32_t num_devices;	//total number of CUDA-capable devices recognized by the host system

		std::vector<CudaDeviceTraits> cuda_device_properties;	//properties of all CUDA-capable devices detected on the host system
		std::vector<unsigned long long> cuda_device_scores;		//stores score for each CUDA-capable device detected on the system

		//The following boolean variables describe the traits of those devices that should be included into the search
		std::pair<uint32_t, uint32_t> minimal_compute_capability;

		bool supports_host_memory_mapping;
		bool supports_concurrent_kernel_execution;
		bool supports_unified_addressing;
		bool supports_stream_priorities;
		bool supports_global_L1_cache;
		bool supports_local_L1_cahce;

		bool is_discrete;
		bool is_ECC_enabled;
		bool is_TCC_driver;
		bool is_multi_gpu;


		bool check_compatibility(uint32_t device_offset) const;	//returns 'true' if device, which has its traits stored at offset "device_offset" meets criteria provided on initialization of the scheduler


		//Performs initialization of the device scheduler
		template<typename ScoreCalculationOpType>
		void initialize(std::pair<uint32_t, uint32_t> minimal_compute_capability, uint32_t flags, ScoreCalculationOpType score_calculation_op)
		{
			//Gather information about CUDA-capable devices available on the host system
			int dev_count;
			cudaGetDeviceCount(&dev_count);
			num_devices = static_cast<uint32_t>(dev_count);

			cuda_device_properties.reserve(num_devices);
			for (unsigned int i = 0; i < num_devices; ++i)
				cuda_device_properties.push_back(CudaDeviceTraits{ i });

			std::sort(cuda_device_properties.begin(), cuda_device_properties.end(),
				[this, &score_calculation_op](const CudaDeviceTraits& dev1, const CudaDeviceTraits& dev2) -> bool
			{
				return score_calculation_op(dev1) > score_calculation_op(dev2);
			}
			);


			for (unsigned int i = 0; i < num_devices; ++i)
				cuda_device_scores.push_back(score_calculation_op(cuda_device_properties[i]));

			//Store minimal requested compute capability
			this->minimal_compute_capability = minimal_compute_capability;

			//Store desired device capabilities
			supports_host_memory_mapping = false;
			supports_concurrent_kernel_execution = false;
			supports_unified_addressing = false;
			supports_stream_priorities = false;
			supports_global_L1_cache = false;
			supports_local_L1_cahce = false;
			is_discrete = false;
			is_ECC_enabled = false;
			is_TCC_driver = false;
			is_multi_gpu = false;

			if ((flags & CDS_SUPPORTS_HOST_MEMORY_MAPPING) == CDS_SUPPORTS_HOST_MEMORY_MAPPING) supports_host_memory_mapping = true;
			if ((flags & CDS_SUPPORTS_CONCURRENT_KERNEL_EXECUTION) == CDS_SUPPORTS_CONCURRENT_KERNEL_EXECUTION) supports_concurrent_kernel_execution = true;
			if ((flags & CDS_SUPPORTS_UNIFIED_ADDRESSING) == CDS_SUPPORTS_UNIFIED_ADDRESSING) supports_unified_addressing = true;
			if ((flags & CDS_SUPPORTS_STREAM_PRIORITIES) == CDS_SUPPORTS_STREAM_PRIORITIES) supports_stream_priorities = true;
			if ((flags & CDS_SUPPORTS_GLOBAL_L1_CACHE) == CDS_SUPPORTS_GLOBAL_L1_CACHE) supports_global_L1_cache = true;
			if ((flags & CDS_SUPPORTS_LOCAL_L1_CACHE) == CDS_SUPPORTS_LOCAL_L1_CACHE) supports_local_L1_cahce = true;

			if ((flags & CDS_DEVICE_DISCRETE) == CDS_DEVICE_DISCRETE) is_discrete = true;
			if ((flags & CDS_DEVICE_ECC_ENABLED) == CDS_DEVICE_ECC_ENABLED) is_ECC_enabled = true;
			if ((flags & CDS_DEVICE_TCC_DRIVER) == CDS_DEVICE_TCC_DRIVER) is_TCC_driver = true;
			if ((flags & CDS_DEVICE_MULTI_GPU) == CDS_DEVICE_MULTI_GPU) is_multi_gpu = true;
		}

		unsigned long long default_score(const CudaDeviceTraits& dev_traits) const;	//default device score calculator


	public:
		typedef CudaDeviceIterator device_iterator;

		//Default initialization. All CUDA-capable devices recognized by the host system should be considered by the scheduler
		CudaDeviceScheduler();

		//Initializes device scheduler using provided settings for minimal desired compute capability and capability flags.
		//The scheduler will consider those devices that have at least requested compute capability and that meet requirements defined by flags.
		CudaDeviceScheduler(std::pair<uint32_t, uint32_t> minimal_compute_capability, uint32_t flags);

		//Initializes device scheduler using provided minimal required compute capability and property flags. In addition preference will be given to
		//those devices that compare "better" in accordance with provided score calculation operation. 
		template<typename ScoreCalculationOpType>
		CudaDeviceScheduler(std::pair<uint32_t, uint32_t> minimal_compute_capability, uint32_t flags, ScoreCalculationOpType score_calculation_op)
		{
			initialize(minimal_compute_capability, flags, score_calculation_op);
		}


		//Returns iterator pointing to the first CUDA-capable device considered by the scheduler
		device_iterator bestDevice();

		//Returns iterator pointing to the last CUDA-capable device considered by the scheduler
		device_iterator worstDevice();

		//Returns iterator pointing to the first CUDA-capable device considered by the scheduler
		device_iterator begin();

		//Returns iterator pointing to the last CUDA-capable device considered by the scheduler
		device_iterator end();

		//Returns quotient representing a part of a distributed task, which should be assigned to the device pointed by provided iterator.
		//Generally speaking, if getLoadFactor() returns 0.5 for device A and 0.25 for device B that means that device A should perform one half of the computations,
		//whilst device B takes care of only one quarter.
		double getLoadFactor(const CudaDeviceIterator& device_iterator) const;

		//Returns number of CUDA-capable devices that comply with requirements provided on initialization of the scheduler
		uint32_t getNumberOfDevices() const;
	};
}


#endif
