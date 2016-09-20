#include "CudaDeviceScheduler.h"

using namespace CudaUtils;


CudaDeviceTraits::CudaDeviceTraits(uint32_t cuda_device_id) : cuda_id(cuda_device_id), error_string(""), error_state(false)
{
	cudaDeviceProp device_properties;
	if (cudaGetDeviceProperties(&device_properties, cuda_device_id) != cudaSuccess)
	{
		error_state = true;
		error_string = "Unable to retrieve properties of device " + std::to_string(cuda_device_id) +
			". The device identifier appears to be invalid";
		return;
	}

	memcpy(name, device_properties.name, 256 * sizeof(char));
	totalGlobalMem = device_properties.totalGlobalMem;
	sharedMemPerBlock = device_properties.sharedMemPerBlock;
	regsPerBlock = device_properties.regsPerBlock;
	warpSize = device_properties.warpSize;
	memPitch = device_properties.memPitch;
	maxThreadsPerBlock = device_properties.maxThreadsPerBlock;
	memcpy(maxThreadsDim, device_properties.maxThreadsDim, sizeof(int) * 3);
	memcpy(maxGridSize, device_properties.maxGridSize, sizeof(int) * 3);
	clockRate = device_properties.clockRate;
	totalConstMem = device_properties.totalConstMem;
	major = device_properties.major;
	minor = device_properties.minor;
	textureAlignment = device_properties.textureAlignment;
	texturePitchAlignment = device_properties.texturePitchAlignment;
	multiProcessorCount = device_properties.multiProcessorCount;
	kernelExecTimeoutEnabled = device_properties.kernelExecTimeoutEnabled;
	integrated = device_properties.integrated;
	canMapHostMemory_ = device_properties.canMapHostMemory;
	computeMode = device_properties.computeMode;
	maxTexture1D = device_properties.maxTexture1D;
	maxTexture1DMipmap = device_properties.maxTexture1DMipmap;
	maxTexture1DLinear = device_properties.maxTexture1DLinear;
	memcpy(maxTexture2D, device_properties.maxTexture2D, sizeof(int) * 2);
	memcpy(maxTexture2DMipmap, device_properties.maxTexture2DMipmap, sizeof(int) * 2);
	memcpy(maxTexture2DLinear, device_properties.maxTexture2DLinear, sizeof(int) * 3);
	memcpy(maxTexture2DGather, device_properties.maxTexture2DGather, sizeof(int) * 2);
	memcpy(maxTexture3D, device_properties.maxTexture3D, sizeof(int) * 3);
	memcpy(maxTexture3DAlt, device_properties.maxTexture3DAlt, sizeof(int) * 3);
	maxTextureCubemap = device_properties.maxTextureCubemap;
	memcpy(maxTexture1DLayered, device_properties.maxTexture1DLayered, sizeof(int) * 2);
	memcpy(maxTexture2DLayered, device_properties.maxTexture2DLayered, sizeof(int) * 3);
	memcpy(maxTextureCubemapLayered, device_properties.maxTextureCubemapLayered, sizeof(int) * 2);
	maxSurface1D = device_properties.maxSurface1D;
	memcpy(maxSurface2D, device_properties.maxSurface2D, sizeof(int) * 2);
	memcpy(maxSurface3D, device_properties.maxSurface3D, sizeof(int) * 3);
	memcpy(maxSurface1DLayered, device_properties.maxSurface1DLayered, sizeof(int) * 2);
	memcpy(maxSurface2DLayered, device_properties.maxSurface2DLayered, sizeof(int) * 3);
	maxSurfaceCubemap = device_properties.maxSurfaceCubemap;
	memcpy(maxSurfaceCubemapLayered, device_properties.maxSurfaceCubemapLayered, sizeof(int) * 2);
	surfaceAlignment = device_properties.surfaceAlignment;
	concurrentKernels = device_properties.concurrentKernels;
	ECCEnabled = device_properties.ECCEnabled;
	pciBusID = device_properties.pciBusID;
	pciDeviceID = device_properties.pciDeviceID;
	pciDomainID = device_properties.pciDomainID;
	tccDriver = device_properties.tccDriver;
	asyncEngineCount = device_properties.asyncEngineCount;
	unifiedAddressing = device_properties.unifiedAddressing;
	memoryClockRate = device_properties.memoryClockRate;
	memoryBusWidth = device_properties.memoryBusWidth;
	l2CacheSize = device_properties.l2CacheSize;
	maxThreadsPerMultiProcessor = device_properties.maxThreadsPerMultiProcessor;
	streamPrioritiesSupported = device_properties.streamPrioritiesSupported;
	globalL1CacheSupported = device_properties.globalL1CacheSupported;
	localL1CacheSupported = device_properties.localL1CacheSupported;
	sharedMemPerMultiProcessor = device_properties.sharedMemPerMultiprocessor;
	regsPerMultiProcessor = device_properties.regsPerMultiprocessor;
	isMultiGpuBoard = device_properties.isMultiGpuBoard;
	multiGpuBoardGroupID = device_properties.multiGpuBoardGroupID;
}

uint32_t CudaDeviceTraits::getCudaDeviceId() const { return cuda_id; }

std::string CudaDeviceTraits::getDeviceName() const { return name; }

size_t CudaDeviceTraits::getTotalGlobalMemory() const { return totalGlobalMem; }

size_t CudaDeviceTraits::getSharedMemoryPerBlock() const { return sharedMemPerBlock; }

int CudaDeviceTraits::getNumberOf32BitRegistersPerBlock() const { return regsPerBlock; }

int CudaDeviceTraits::getWarpSize() const { return warpSize; }

size_t CudaDeviceTraits::getMemoryPitchSize() const { return memPitch; }

int CudaDeviceTraits::getNumberOfThreadsPerBlock() const { return maxThreadsPerBlock; }

dim3 CudaDeviceTraits::getNumberOfThreadsPerBlockDimension() const 
{ 
	return dim3(maxThreadsDim[0], maxThreadsDim[1], maxThreadsDim[2]);
}

dim3 CudaDeviceTraits::getNumberOfBlocksInGrid() const
{
	return dim3(maxGridSize[0], maxGridSize[1], maxGridSize[2]);
}

uint32_t CudaDeviceTraits::getGpuClockRate() const { return clockRate; }

size_t CudaDeviceTraits::getConstantMemory() const { return totalConstMem; }

std::pair<uint32_t, uint32_t> CudaDeviceTraits::getComputeCapability() const
{
	return std::make_pair(major, minor);
}

size_t CudaDeviceTraits::getTextureAlignment() const { return textureAlignment; }

size_t CudaDeviceTraits::getTexturePitchAlignment() const { return texturePitchAlignment; }

uint32_t CudaDeviceTraits::getNumberOfMultiprocessors() const { return multiProcessorCount; }

bool CudaDeviceTraits::isKernelExecutionTimeoutEnabled() const { return kernelExecTimeoutEnabled > 0; }

bool CudaDeviceTraits::isIntegrated() const { return integrated > 0; }

bool CudaDeviceTraits::canMapHostMemory() const { return canMapHostMemory_ > 0; }

uint32_t CudaDeviceTraits::getComputeMode() const { return computeMode; }

uint32_t CudaDeviceTraits::getMaximum1DTextureSize() const { return maxTexture1D; }

uint32_t CudaDeviceTraits::getMaximum1DTextureMipmapSize() const { return maxTexture1DMipmap; }

uint32_t CudaDeviceTraits::getMaximum1DTextureLinearSize() const { return maxTexture1DLinear; }

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum2DTextureSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxTexture2D[0]), static_cast<uint32_t>(maxTexture2D[1])};
	return rv;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum2DTextureMipmapSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxTexture2DMipmap[0]), static_cast<uint32_t>(maxTexture2DMipmap[1])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum2DTextureLinearSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxTexture2DLinear[0]), static_cast<uint32_t>(maxTexture2DLinear[1]), static_cast<uint32_t>(maxTexture2DLinear[2])};
	return rv;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum2DTextureGatherSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxTexture2DGather[0]), static_cast<uint32_t>(maxTexture2DGather[1])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum3DTextureSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxTexture3D[0]), static_cast<uint32_t>(maxTexture3D[1]), static_cast<uint32_t>(maxTexture3D[2])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum3DTextureAlternateSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxTexture3DAlt[0]), static_cast<uint32_t>(maxTexture3DAlt[1]), static_cast<uint32_t>(maxTexture3DAlt[2])};
	return rv;
}

uint32_t CudaDeviceTraits::getMaximumCubemapTextureSize() const
{
	return maxTextureCubemap;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum1DLayeredTextureSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxTexture1DLayered[0]), static_cast<uint32_t>(maxTexture1DLayered[1])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum2DLayeredTextureSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxTexture2DLayered[0]), static_cast<uint32_t>(maxTexture2DLayered[1]), static_cast<uint32_t>(maxTexture2DLayered[2])};
	return rv;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximumCubemapLayeredTextureSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxTextureCubemapLayered[0]), static_cast<uint32_t>(maxTextureCubemapLayered[1])};
	return rv;
}

uint32_t CudaDeviceTraits::getMaximum1DSurfaceSize() const
{
	return maxSurface1D;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum2DSurfaceSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxSurface2D[0]), static_cast<uint32_t>(maxSurface2D[1])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum3DSurfaceSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxSurface3D[0]), static_cast<uint32_t>(maxSurface3D[1]), static_cast<uint32_t>(maxSurface3D[2])};
	return rv;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximum1DLayeredSurfaceSize() const
{
	std::array < uint32_t, 2 > rv = {static_cast<uint32_t>(maxSurface1DLayered[0]), static_cast<uint32_t>(maxSurface1DLayered[1])};
	return rv;
}

std::array<uint32_t, 3> CudaDeviceTraits::getMaximum2DLayeredSurfaceSize() const
{
	std::array < uint32_t, 3 > rv = {static_cast<uint32_t>(maxSurface2DLayered[0]), static_cast<uint32_t>(maxSurface2DLayered[1]), static_cast<uint32_t>(maxSurface2DLayered[2])};
	return rv;
}

uint32_t CudaDeviceTraits::getMaximumSurfaceCubemapSize() const
{
	return maxSurfaceCubemap;
}

std::array<uint32_t, 2> CudaDeviceTraits::getMaximumSurfaceCubemapLayeredSize() const
{
	std::array<uint32_t, 2> rv = { static_cast<uint32_t>(maxSurfaceCubemapLayered[0]), static_cast<uint32_t>(maxSurfaceCubemapLayered[1]) };
	return rv;
}

size_t CudaDeviceTraits::getSurfaceAlignmentRequirement() const
{
	return surfaceAlignment;
}

bool CudaDeviceTraits::supportsConcurrentKernelExecution() const { return concurrentKernels > 0; }

bool CudaDeviceTraits::isECCEnabled() const { return ECCEnabled > 0; }

int CudaDeviceTraits::getPCIBusId() const { return pciBusID; }

int CudaDeviceTraits::getPCIDeviceId() const { return pciDeviceID; }

int CudaDeviceTraits::getPCIDomainId() const { return pciDeviceID; }

bool CudaDeviceTraits::hasTCCDriver() const { return tccDriver > 0; }

uint32_t CudaDeviceTraits::getNumberOfAsynchronousEngines() const
{
	return asyncEngineCount;
}

bool CudaDeviceTraits::supportsUnifiedAddressSpace() const
{
	return unifiedAddressing > 0;
}

uint32_t CudaDeviceTraits::getPeakMemoryClockRate() const
{
	return memoryClockRate;
}

uint32_t CudaDeviceTraits::getGlobalMemoryBusWidth() const
{
	return memoryBusWidth;
}

size_t CudaDeviceTraits::getLevel2CacheMemorySize() const
{
	return l2CacheSize;
}

size_t CudaDeviceTraits::getMaximumThreadsPerMultiprocessor() const
{
	return maxThreadsPerMultiProcessor;
}

bool CudaDeviceTraits::supportsStreamPriorities() const
{
	return streamPrioritiesSupported > 0;
}

bool CudaDeviceTraits::supportsL1GlobalCache() const
{
	return globalL1CacheSupported > 0;
}

bool CudaDeviceTraits::supportsL1LocalCache() const
{
	return localL1CacheSupported > 0;
}

size_t CudaDeviceTraits::getSharedMemoryPerMultiprocessor() const
{
	return sharedMemPerMultiProcessor;
}

uint32_t CudaDeviceTraits::get32BitRegistersPerMultiprocessor() const
{
	return regsPerMultiProcessor;
}

bool CudaDeviceTraits::isMultiGPU() const { return isMultiGpuBoard > 0; }

int CudaDeviceTraits::getMultiGpuBoardIdentifier() const { return multiGpuBoardGroupID; }



CudaDeviceTraits::operator bool() const { return !error_state; }

const char* CudaDeviceTraits::getErrorString() const { return error_string.c_str(); }





CudaDeviceScheduler::CudaDeviceIterator::CudaDeviceIterator(CudaDeviceScheduler& owner) : p_owner{ &owner }
{
	device_offset = 0;
	while (device_offset < static_cast<int>(p_owner->cuda_device_properties.size()) && !p_owner->check_compatibility(device_offset)) ++device_offset;
}

CudaDeviceScheduler::CudaDeviceIterator::CudaDeviceIterator(CudaDeviceScheduler& owner, int) : p_owner{ &owner }
{
	device_offset = static_cast<int>(p_owner->cuda_device_properties.size());
}

CudaDeviceScheduler::CudaDeviceIterator::CudaDeviceIterator(const CudaDeviceScheduler::CudaDeviceIterator& other) : device_offset{ other.device_offset }, p_owner{ other.p_owner }
{

}

CudaDeviceScheduler::CudaDeviceIterator& CudaDeviceScheduler::CudaDeviceIterator::operator=(const CudaDeviceScheduler::CudaDeviceIterator& other)
{
	device_offset = other.device_offset;
	p_owner = other.p_owner;
	return *this;
}

CudaDeviceTraits CudaDeviceScheduler::CudaDeviceIterator::operator*() const
{
	return p_owner->cuda_device_properties[device_offset];
}

CudaDeviceTraits* CudaDeviceScheduler::CudaDeviceIterator::operator->() const
{
	return &(p_owner->cuda_device_properties[device_offset]);
}

CudaDeviceScheduler::CudaDeviceIterator& CudaDeviceScheduler::CudaDeviceIterator::operator++()
{
	device_offset = device_offset < static_cast<int>(p_owner->cuda_device_properties.size()) ? device_offset + 1 : device_offset;
	while (device_offset < static_cast<int>(p_owner->cuda_device_properties.size()) && !p_owner->check_compatibility(device_offset)) ++device_offset;

	return *this;
}

CudaDeviceScheduler::CudaDeviceIterator CudaDeviceScheduler::CudaDeviceIterator::operator++(int)
{
	CudaDeviceIterator old_iterator_copy{ *this };

	++(*this);

	return old_iterator_copy;
}

bool CudaDeviceScheduler::CudaDeviceIterator::operator==(const CudaDeviceScheduler::CudaDeviceIterator& other) const
{
	return p_owner == other.p_owner && device_offset == other.device_offset;
}

bool CudaDeviceScheduler::CudaDeviceIterator::operator!=(const CudaDeviceScheduler::CudaDeviceIterator& other) const
{
	return !(*this == other);
}

CudaDeviceScheduler::CudaDeviceIterator& CudaDeviceScheduler::CudaDeviceIterator::operator--()
{
	device_offset = device_offset >= 0 ? device_offset - 1 : device_offset;
	while (device_offset >= 0 && !p_owner->check_compatibility(device_offset)) --device_offset;
	
	return *this;
}


CudaDeviceScheduler::CudaDeviceIterator CudaDeviceScheduler::CudaDeviceIterator::operator--(int)
{
	CudaDeviceIterator old_iterator_copy{ *this };

	--(*this);

	return old_iterator_copy;
}



bool CudaDeviceScheduler::check_compatibility(uint32_t device_offset) const
{
	if (minimal_compute_capability.first > cuda_device_properties[device_offset].getComputeCapability().first) return false;
	if (minimal_compute_capability.first == cuda_device_properties[device_offset].getComputeCapability().first &&
		minimal_compute_capability.second > cuda_device_properties[device_offset].getComputeCapability().second) return false;

	if (supports_host_memory_mapping && !cuda_device_properties[device_offset].canMapHostMemory()) return false;
	if (supports_concurrent_kernel_execution && !cuda_device_properties[device_offset].supportsConcurrentKernelExecution()) return false;
	if (supports_unified_addressing && !cuda_device_properties[device_offset].supportsUnifiedAddressSpace()) return false;
	if (supports_stream_priorities && !cuda_device_properties[device_offset].supportsStreamPriorities()) return false;
	if (supports_global_L1_cache && !cuda_device_properties[device_offset].supportsL1GlobalCache()) return false;
	if (supports_local_L1_cahce && !cuda_device_properties[device_offset].supportsL1LocalCache()) return false;
	if (is_discrete && cuda_device_properties[device_offset].isIntegrated()) return false;
	if (is_ECC_enabled && !cuda_device_properties[device_offset].isECCEnabled()) return false;
	if (is_TCC_driver && !cuda_device_properties[device_offset].hasTCCDriver()) return false;
	if (is_multi_gpu && !cuda_device_properties[device_offset].isMultiGPU()) return false;

	return true;
}

unsigned long long CudaDeviceScheduler::default_score(const CudaDeviceTraits& dev_traits) const
{
	//Default device score is a product of memory bandwidth, GPU clock, and memory size

	double GPUclock = dev_traits.getGpuClockRate() / 1000.0;	//GPU clock in MHz
	double MemoryBandwidth = (dev_traits.getGlobalMemoryBusWidth() / 8.0) * (dev_traits.getPeakMemoryClockRate() / 1000.0);	//Global memory bandwidth in MBs/sec (1MB = 1000000bytes)
	double MemoryCapacity = dev_traits.getTotalGlobalMemory() / (1024.0 * 1024.0 * 1024.0);	//Total global memory capacity represented in gigabytes
	int num_multiprocessors = dev_traits.getNumberOfMultiprocessors();	//Number of multiprocessors provided by the device
	int num_async_engines = dev_traits.getNumberOfAsynchronousEngines();	//Number of asynchronous engines

	double score = GPUclock * MemoryBandwidth * MemoryCapacity * num_multiprocessors * num_async_engines;
	return static_cast<unsigned long long>(std::round(score));
}


CudaDeviceScheduler::CudaDeviceScheduler()
{
	initialize(std::make_pair(1U, 0U), 0, std::function < unsigned long long(const CudaDeviceTraits&) > {std::bind(&CudaDeviceScheduler::default_score, *this, std::placeholders::_1)});
}

CudaDeviceScheduler::CudaDeviceScheduler(std::pair<uint32_t, uint32_t> minimal_compute_capability, uint32_t flags)
{
	initialize(minimal_compute_capability, flags, std::function < unsigned long long(const CudaDeviceTraits&) > {std::bind(&CudaDeviceScheduler::default_score, *this, std::placeholders::_1)});
}

CudaDeviceScheduler::device_iterator CudaDeviceScheduler::bestDevice()
{
	return CudaDeviceIterator{ *this };
}

CudaDeviceScheduler::device_iterator CudaDeviceScheduler::worstDevice()
{
	return (--CudaDeviceIterator{ *this, 0 });
}

CudaDeviceScheduler::device_iterator CudaDeviceScheduler::begin()
{
	return CudaDeviceIterator{ *this };
}

CudaDeviceScheduler::device_iterator CudaDeviceScheduler::end()
{
	return CudaDeviceIterator{ *this, 0 };
}

double CudaDeviceScheduler::getLoadFactor(const CudaDeviceIterator& device_iterator) const
{
	//Check if provided iterator refers to this scheduler object
	if (device_iterator.p_owner != this)
		throw(std::out_of_range{ "getLoadFactor() error: provided iterator refers to CudeDeviceScheduler object that differs from the context" });

	double full_compute_power = std::accumulate(cuda_device_scores.begin(), cuda_device_scores.end(), 0.0);

	return cuda_device_scores[device_iterator.device_offset] / full_compute_power;
}

uint32_t CudaDeviceScheduler::getNumberOfDevices() const
{
	uint32_t rv = 0;
	for (int dev_offset = 0; dev_offset < static_cast<int>(cuda_device_properties.size()); ++dev_offset)
		if (check_compatibility(dev_offset))++rv;
	return rv;
}




