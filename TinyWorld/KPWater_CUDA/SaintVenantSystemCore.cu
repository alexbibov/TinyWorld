#include "SaintVenantSystemCore.cuh"
#include "CudaErrorWrapper.h"
#include "CudaDeviceScheduler.h"

#pragma comment(lib,"cudart")

using namespace SaintVenantSystem;
using namespace CudaUtils;


//********************************************************CUDA GPGPU function declarations**********************************************************
//Low-level device functions

//Computes minmod nonlinear delimiter for arguments a1, a2 and a3
//minmod(a1,a2,a3)=max(a1,a2,a3) if all arguments are less then zero,
//minmod(a1,a2,a3)=min(a1,a2,a3) if all arguments are greater then zero,
//minmod(a1,a2,a3)=0 otherwise.
__device__ Numeric __minmod__(Numeric a1, Numeric a2, Numeric a3);

//Computes horizontal flux component F of the Saint-Venant system as defined by the KP-scheme
__device__ Numeric3 __flux_F__(Numeric4 T, Numeric B, Numeric g);

//Computes vertical flux component G of the Saint-Venant system as defined by the KP-scheme
__device__ Numeric3 __flux_G__(Numeric4 T, Numeric B, Numeric g);

//Computes horizontal numerical flux
__device__ Numeric3 __flux_Hx__(Numeric4 _Te, Numeric4 _Tw, Numeric g);

//Computes vertical numerical flux
__device__ Numeric3 __flux_Hy__(Numeric4 _Tn, Numeric4 _Ts, Numeric g);

//Extracts grid element based on a block-offset address
template<typename block_type>
__device__ const block_type __get_block_element__(const block_type *grid, int block_x, int block_y, int x, int y, int stride);

//Sets value to a given grid node defined by a block-offset address
template<typename block_type>
__device__ void __set_block_element__(block_type *grid, int block_x, int block_y, int x, int y, int stride, block_type element_value);

//Return pointer to an integer-index topography element addressed by a block-offset tuple
__device__ const Numeric* __get_topography_block_element_addr__(const Numeric* B, int block_x, int block_y, int x, int y, int grid_width);


//CUDA Kernels

//Computes cardinal direction components of the water elevation and related fluxes
__global__  void __compute_cardinals__(Numeric4* Te, Numeric4* Tw, Numeric4* Tn, Numeric4* Ts, const Numeric4* U, const Numeric* B,
	Numeric dx, Numeric dy, unsigned int grid_width, unsigned int grid_height, Numeric theta, Numeric eps);

//Computes the right-hand side of the ODE system defined by Kurganov-Petrova scheme
__global__ void __compute_rhs__(Numeric4* rhs, const Numeric4* Te, const Numeric4* Tw, const Numeric4* Tn, const Numeric4* Ts,
	const Numeric4* U, const Numeric* B, Numeric dx, Numeric dy, Numeric g, unsigned int grid_width, unsigned int grid_height);

//Sets east-of-west and west-of-east walls
__global__ void __set_eow_woe_bc__(Numeric4* Te, Numeric4* Tw, const Numeric4* eow, const Numeric4* woe, const Numeric* B, Numeric eps,
	unsigned int grid_width, unsigned int grid_height);

//Sets north-of-south and south-of-north walls
__global__ void __set_nos_son_bc__(Numeric4* Tn, Numeric4* Ts, const Numeric4* nos, const Numeric4* son, const Numeric* B, Numeric eps,
	unsigned int grid_width, unsigned int grid_height, bool reconstructed_nos, bool reconstructed_son);

//Sets boundary values on the west and east walls for subtask chunk U
__global__ void __set_west_east_bc__(Numeric4 *U, const Numeric4* wb, const Numeric4* eb, unsigned int grid_height, unsigned int grid_width);

//Sets boundary values on the south wall for subtask chunk U
__global__ void __set_south_bc__(Numeric4 *U, const Numeric4* sb, unsigned int grid_width);

//Sets boundary values on the north wall for subtask chunk U
__global__ void __set_north_bc__(Numeric4 *U, const Numeric4* nb, unsigned int grid_height, unsigned int grid_width);

//Implements CUDA-kernel for Euler solver
__global__ void __euler__(Numeric4* U0, const Numeric4* rhs_U0, Numeric dt, unsigned int grid_height, unsigned int grid_width);

//Implements second semi-step of SSP-RK(2,2)
__global__ void __ssprk22_2__(Numeric4* U1, const Numeric4* U0, const Numeric4* rhs_U1, Numeric dt, unsigned int grid_height, unsigned int grid_width);

//Implements second semi-step for SSP-RK(3,3)
__global__ void __ssprk33_2__(Numeric4* U1, const Numeric4* U0, const Numeric4* rhs_U1, Numeric dt, unsigned int grid_height, unsigned int grid_width);

//Implements third semi-step for SSP-RK(3,3)
__global__ void __ssprk33_3__(Numeric4* U2, const Numeric4* U0, const Numeric4* rhs_U2, Numeric dt, unsigned int grid_height, unsigned int grid_width);
//*************************************************************************************************************************************************


SVSCore::SVSCore() : 

cpu_system_state{ nullptr }, cpu_wb{ nullptr }, cpu_eb{ nullptr }, cpu_sb{ nullptr }, cpu_nb{ nullptr },
cpu_eow{ nullptr }, cpu_woe{ nullptr }, cpu_nos{ nullptr }, cpu_son{ nullptr }, cpu_bc_exchange_buf{ nullptr }, 
CUDA_device_scheduler{ std::make_pair(2U, 0U), CDS_DEVICE_DISCRETE }, CUDA_device_count{ 0 },
best_device_id{ CUDA_device_scheduler.bestDevice()->getCudaDeviceId() },
worst_device_id{ CUDA_device_scheduler.worstDevice()->getCudaDeviceId() }, 
error_state{ false }, error_source{ -1 }, 
error_callback{ [](const CudaErrorWrapper& err, int dev_id) -> void{} },
is_initialized{ false }, exec_time{ 0.0f }

{
	reset();
}


SVSCore::SVSCore(const SVSCore& other) :
CUDA_device_scheduler{ other.CUDA_device_scheduler }, CUDA_device_count{ other.CUDA_device_count }, error_descriptor{ other.error_descriptor },
error_state{ other.error_state }, error_callback{ other.error_callback }, error_source{ other.error_source }, is_initialized{ other.is_initialized }, 
memCPU{ other.memCPU }, memGPU(other.memGPU), best_device_id{ other.best_device_id }, worst_device_id{ other.worst_device_id }
{
	//Initialize vector of GPU memory blocks
	//memGPU = GPUMemoryBlock_Vector(CUDA_device_count);

	//Initialize device memory pointers
	gpu_system_state = Numeric4pointer_Vector(CUDA_device_count);
	gpu_temp = Numeric4pointer_Vector(CUDA_device_count);
	gpu_rhs = Numeric4pointer_Vector(CUDA_device_count);
	gpu_Te = Numeric4pointer_Vector(CUDA_device_count); gpu_Tw = Numeric4pointer_Vector(CUDA_device_count);
	gpu_Ts = Numeric4pointer_Vector(CUDA_device_count); gpu_Tn = Numeric4pointer_Vector(CUDA_device_count);
	gpu_eow = Numeric4pointer_Vector(CUDA_device_count); gpu_woe = Numeric4pointer_Vector(CUDA_device_count);
	gpu_nos = Numeric4pointer_Vector(CUDA_device_count); gpu_son = Numeric4pointer_Vector(CUDA_device_count);
	gpu_wb = Numeric4pointer_Vector(CUDA_device_count); gpu_eb = Numeric4pointer_Vector(CUDA_device_count);
	gpu_B = Numericpointer_Vector(CUDA_device_count);

	//If the source object has been initialized, copy the SVS-options and memory buffers.
	//Some objects related to the CUDA context (like CUDA streams and events should be re-created for this object instance to guarantee their lifetime)
	if (other.is_initialized)
	{
		st_grid_height = other.st_grid_height;
		st_dev_stream = cudaStream_t_Vector(other.st_dev_stream.size());	//Device streams should be re-instantiated to guarantee their existance thoughout the lifetime of the object
		st_ctxc = cudaEvent_t_Vector(other.st_ctxc.size());	//CUDA events should be redefined to guarantee their lifetime
		st_start = cudaEvent_t_Vector(other.st_start.size());
		st_finish = cudaEvent_t_Vector(other.st_finish.size());
		opts = other.opts;
		exec_time = other.exec_time;
		//size_t taskCPUMemCost = sizeof(Numeric4)*((opts.grid_width + 2)*(opts.grid_height + 2) + 4 * opts.grid_height + 5 * opts.grid_width);
		//memCPU = other.memCPU;

		//Initialize CPU memory segments
		//The allocations made below are "fake" allocations as all the meaningful data have already been copied to the destination buffer, thus
		//we only need to compute the appropriate offsets to the allocation chunks
		memCPU.reset();	//ensure that "allocations" begin from the zero offset
		cpu_system_state = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*(opts.grid_width + 2)*(opts.grid_height + 2)));
		cpu_wb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_eb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_eow = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_woe = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_sb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_nb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_nos = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_son = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_bc_exchange_buf = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cudaHostRegister(cpu_bc_exchange_buf, sizeof(Numeric4)*opts.grid_width, cudaHostRegisterPortable);


		for (auto device : CUDA_device_scheduler)
		{
			//Retrieve numerical identifier of the currently enumerated device
			uint32_t id = device.getCudaDeviceId();
			cudaSetDevice(id);


			//Create stream object and events for this device
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaStreamCreate(&st_dev_stream[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_ctxc[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_start[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_finish[id]))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}


			//Compute global memory cost of the subtask in bytes
			/*const unsigned int st_GPUGlblMemCost =
				sizeof(Numeric4)*(2 * (opts.grid_width + 2)*(st_grid_height[id] + 2) +    //system state and temporary buffer
				2 * (opts.grid_width + 1)*st_grid_height[id] +    //Te and Tw
				2 * opts.grid_width*(st_grid_height[id] + 1) +    //Tn and Ts
				2 * st_grid_height[id] +    //east-of-west and west-of-east boundary values
				2 * opts.grid_width +    //north-of-south and south-of-north boundary value
				2 * st_grid_height[id] +	//west and east boundary values
				opts.grid_width*st_grid_height[id] +	//rhs
				(id == best_device_id ? opts.grid_width : 0) +	//first device allocates memory for southern wall
				(id == worst_device_id ? opts.grid_width : 0)) +	//last device allocates memory for northern wall
				sizeof(Numeric)*(2 * opts.grid_width + 1)*(2 * st_grid_height[id] + 1);    //B	*/
			//memGPU[id] = other.memGPU[id];

			//Allocate memory on the current device
			//The allocations made below are "fake" allocations as all the meaningful data have already been copied to the destination buffer, thus
			//we only need to compute the appropriate offsets to the allocation chunks
			memGPU[id].reset();	//ensure that "allocations" begin from the zero offset
			gpu_system_state[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
			gpu_temp[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
			gpu_Te[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
			gpu_Tw[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
			gpu_Ts[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
			gpu_Tn[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
			gpu_rhs[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * opts.grid_width));
			gpu_eow[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_woe[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_nos[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
			gpu_son[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
			gpu_wb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_eb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));

			if (id == best_device_id)	//first device allocates memory for southern wall
				gpu_sb = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));

			if (id == worst_device_id)	//last device allocates memory for northern wall
				gpu_nb = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));

			gpu_B[id] = static_cast<Numeric*>(memGPU[id].allocate(sizeof(Numeric)*(2 * st_grid_height[id] + 1)*(2 * opts.grid_width + 1)));

			//Copy initial state to the GPU memory
			//cudaMemcpyAsync(gpu_system_state[id], other.gpu_system_state[id], sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2),
			//	cudaMemcpyDeviceToDevice, st_dev_stream[id]);

			//Copy topography component to the GPU memory
			//cudaMemcpyAsync(gpu_B[id], other.gpu_B[id], sizeof(Numeric)*(2 * opts.grid_width + 1)*(2 * st_grid_height[id] + 1),
			//	cudaMemcpyDeviceToDevice, st_dev_stream[id]);

			/*if ((error_descriptor = CudaErrorWrapper{ cudaPeekAtLastError(), __FILE__, __func__, __LINE__ }))
			{
				error_state = true;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}*/
		}
	}
}


SVSCore& SVSCore::operator=(const SVSCore& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Copy the state of the object
	if (other.is_initialized)
	{
		st_grid_height = other.st_grid_height;
		st_dev_stream = cudaStream_t_Vector(other.st_dev_stream.size());	//Device streams should be re-instantiated to guarantee their existance thoughout the lifetime of the object
		st_ctxc = cudaEvent_t_Vector(other.st_ctxc.size());	//CUDA events should be redefined to guarantee their lifetime
		st_start = cudaEvent_t_Vector(other.st_start.size());
		st_finish = cudaEvent_t_Vector(other.st_finish.size());
		opts = other.opts;
		exec_time = other.exec_time;

		//Copy memory buffers
		memGPU = other.memGPU;
		memCPU = other.memCPU;


		//Setup host pointers
		//The allocations made below are "fake" allocations as all the meaningful data have already been copied to the destination buffer, thus
		//we only need to compute the appropriate offsets to the allocation chunks
		memCPU.reset();		//reset allocation caret of the CPU buffer to zero
		cpu_system_state = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*(opts.grid_width + 2)*(opts.grid_height + 2)));
		cpu_wb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_eb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_eow = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_woe = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
		cpu_sb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_nb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_nos = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_son = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cpu_bc_exchange_buf = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
		cudaHostRegister(cpu_bc_exchange_buf, sizeof(Numeric4)*opts.grid_width, cudaHostRegisterPortable);


		//Setup GPU pointers
		for (auto device : CUDA_device_scheduler)
		{
			//Retrieve numerical identifier of the currently enumerated device
			uint32_t id = device.getCudaDeviceId();


			//Create stream object and events for this device
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaStreamCreate(&st_dev_stream[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_ctxc[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_start[id]))) ||
				(aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_finish[id]))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return *this;
			}


			//Setup device memory pointers
			//The allocations made below are "fake" allocations as all the meaningful data have already been copied to the destination buffer, thus
			//we only need to compute the appropriate offsets to the allocation chunks
			memGPU[id].reset();	//reset allocation caret of the currently processed GPU buffer to zero
			gpu_system_state[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
			gpu_temp[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
			gpu_Te[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
			gpu_Tw[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
			gpu_Ts[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
			gpu_Tn[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
			gpu_rhs[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * opts.grid_width));
			gpu_eow[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_woe[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_nos[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
			gpu_son[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
			gpu_wb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
			gpu_eb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));

			if (id == best_device_id)	//first device allocates memory for southern wall
				gpu_sb = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));

			if (id == worst_device_id)	//last device allocates memory for northern wall
				gpu_nb = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));

			gpu_B[id] = static_cast<Numeric*>(memGPU[id].allocate(sizeof(Numeric)*(2 * st_grid_height[id] + 1)*(2 * opts.grid_width + 1)));
		}
	}

	error_descriptor = other.error_descriptor;
	error_state = other.error_state;
	error_source = other.error_source;
	error_callback = other.error_callback;

	return *this;
}


SVSCore::SVSCore(const SVSVarU init_state, unsigned int height, unsigned int width, 
	Numeric dx, Numeric dy, Numeric g, Numeric theta, Numeric eps, const Numeric *topography) : 

	cpu_system_state{ nullptr }, cpu_wb{ nullptr }, cpu_eb{ nullptr }, cpu_sb{ nullptr }, cpu_nb{ nullptr },
	cpu_eow{ nullptr }, cpu_woe{ nullptr }, cpu_nos{ nullptr }, cpu_son{ nullptr }, cpu_bc_exchange_buf{ nullptr },
	CUDA_device_scheduler{ std::make_pair(2U, 0U), CDS_DEVICE_DISCRETE }, CUDA_device_count{ 0 },
	best_device_id{ CUDA_device_scheduler.bestDevice()->getCudaDeviceId() },
	worst_device_id{ CUDA_device_scheduler.worstDevice()->getCudaDeviceId() }, 
	error_state{ false }, error_callback{ [](const CudaErrorWrapper& err, int dev_id)->void{} }, error_source{ -1 },
	is_initialized{ false }, exec_time{ 0.0f }

{
	reset();
	initialize(init_state,height,width,dx,dy,g,theta,eps,topography);
}


SVSCore::SVSCore(const SVSVarU init_state, SVSParameters opts) :

cpu_system_state{ nullptr }, cpu_wb{ nullptr }, cpu_eb{ nullptr }, cpu_sb{ nullptr }, cpu_nb{ nullptr },
cpu_eow{ nullptr }, cpu_woe{ nullptr }, cpu_nos{ nullptr }, cpu_son{ nullptr }, cpu_bc_exchange_buf{ nullptr },
CUDA_device_scheduler{ std::make_pair(2U, 0U), CDS_DEVICE_DISCRETE }, CUDA_device_count{ 0 },
best_device_id{ CUDA_device_scheduler.bestDevice()->getCudaDeviceId() },
worst_device_id{ CUDA_device_scheduler.worstDevice()->getCudaDeviceId() },
error_state{ false }, error_callback{ [](const CudaErrorWrapper& err, int dev_id)->void{} }, error_source{ -1 },
is_initialized{ false }, exec_time{ 0.0f }

{
	reset();
	initialize(init_state, opts);
}


SVSCore::~SVSCore()
{
	//Release CUDA resources
	if(is_initialized)
	{
		for(auto device : CUDA_device_scheduler)
		{
			//Retrieve numerical identifier of the currently enumerated device
			uint32_t id = device.getCudaDeviceId();

			cudaSetDevice(id);	//set current device
			cudaEventDestroy(st_ctxc[id]);	//destroy context synchronization event
			cudaEventDestroy(st_start[id]);	//destroy GPU performance counter start event
			cudaEventDestroy(st_finish[id]);	//destroy GPU performance counter finish event
			cudaStreamDestroy(st_dev_stream[id]);	//destroy the main command stream
		}
		cudaHostUnregister(cpu_bc_exchange_buf);
	}
}


SVSCore::operator bool() const { return !error_state; }


void SVSCore::reset()
{
	error_descriptor = CudaErrorWrapper{};	//by default there is no last error
	error_state = false;	//reset error state of the object
	error_source = -1;    //by default the error source is not defined
	exec_time = 0.0f;


	//Check if the object has been previously initialized
	if (is_initialized)
	{
		memCPU.release();	//release CPU memory
		for (auto device : CUDA_device_scheduler)
			memGPU[device.getCudaDeviceId()].release();	//release GPU memory for each CUDA-capable GPU
	}
	else
	{
		//Get number of CUDA-capable devices installed on the host system
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaGetDeviceCount(&CUDA_device_count))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = -1;
			error_callback(error_descriptor, error_source);
			return;
		}
	}


	
	//NOTE: The following test for the preprocessor definitions is motivated by cudaDeviceReset() malfunctioning 
	//on Windows operating systems when called from MATLAB MEX module
#if !(defined _WIN32 || defined _WIN64)
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve identifier of the currently processed device
		uint32_t id = device.getCudaDeviceId();
		cudaSetDevice(id);
		cudaDeviceReset();	//Reset the current device

		if ((aux_err_desc = CudaErrorWrapper{ cudaPeekAtLastError(), __FILE__, __func__, __LINE__ }))	//Check if device has been reset correctly
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}
			
	}
#endif



	error_callback = [](const CudaErrorWrapper& err, int dev_id)->void{};	//reset error callback function
	is_initialized = false;	//by default the object is not initialized

	//memGPU = GPUMemoryBlock_Vector(CUDA_device_count);	//initialize vector of GPU memory blocks
	st_grid_height = unsigned_int_Vector(CUDA_device_count);	//initialize vector of subtask chunk sizes
	st_dev_stream = cudaStream_t_Vector(CUDA_device_count);	//initialize vector of device streams

	//Initialize CUDA event vectors
	st_ctxc = cudaEvent_t_Vector(CUDA_device_count);	//kernel synchronization events
	st_start = cudaEvent_t_Vector(CUDA_device_count);	//start timer events
	st_finish = cudaEvent_t_Vector(CUDA_device_count);	//stop timer events

	//Initialize device memory pointers
	gpu_system_state = Numeric4pointer_Vector(CUDA_device_count);
	gpu_temp = Numeric4pointer_Vector(CUDA_device_count);
	gpu_rhs = Numeric4pointer_Vector(CUDA_device_count);
	gpu_Te = Numeric4pointer_Vector(CUDA_device_count); gpu_Tw = Numeric4pointer_Vector(CUDA_device_count);
	gpu_Ts = Numeric4pointer_Vector(CUDA_device_count); gpu_Tn = Numeric4pointer_Vector(CUDA_device_count);
	gpu_eow = Numeric4pointer_Vector(CUDA_device_count); gpu_woe = Numeric4pointer_Vector(CUDA_device_count);
	gpu_nos = Numeric4pointer_Vector(CUDA_device_count); gpu_son = Numeric4pointer_Vector(CUDA_device_count);
	gpu_wb = Numeric4pointer_Vector(CUDA_device_count); gpu_eb = Numeric4pointer_Vector(CUDA_device_count);
	gpu_B = Numericpointer_Vector(CUDA_device_count);
}


float SVSCore::getCudaExecutionTime() const
{
	return is_initialized ? exec_time : 0.0f;
}


std::pair<CudaErrorWrapper, int> SVSCore::getLastError() const
{
	return std::make_pair(error_descriptor, error_source);
}


bool SVSCore::getErrorState() const { return error_state; }


bool SVSCore::isInitialized() const { return is_initialized; }


void SVSCore::registerErrorCallback(const std::function<void(const CudaUtils::CudaErrorWrapper&, int)>& error_callback)
{
	this->error_callback = error_callback;
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve identifier of the currently processed device
		uint32_t id = device.getCudaDeviceId();

		//Register error handler to each of the GPU buffers
		memGPU[id].registerErrorCallback(error_callback);
	}
}


void SVSCore::initialize(const SVSVarU init_state, unsigned int height, unsigned int width, Numeric dx, Numeric dy, Numeric g, Numeric theta, Numeric eps, const Numeric *topography)
{
	//If object has already been  initialized or is in an erroneous state, do nothing and exit
	if (is_initialized || error_state) return;

	//Fill in the system's options
	opts.grid_height = height;
	opts.grid_width = width;
	opts.dx = dx;
	opts.dy = dy;
	opts.g = g;
	opts.theta = theta;
	opts.eps = eps;
	opts.B = topography;

	//Allocate CPU linear memory buffer to store the system state and boundary conditions
	size_t taskCPUMemCost = sizeof(Numeric4)*((opts.grid_width + 2)*(opts.grid_height + 2) + 4 * opts.grid_height + 5 * opts.grid_width);
	memCPU.initialize(taskCPUMemCost);

	//Initialize CPU memory segments
	cpu_system_state = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*(opts.grid_width + 2)*(opts.grid_height + 2)));
	cpu_wb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
	cpu_eb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
	cpu_eow = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
	cpu_woe = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_height));
	cpu_sb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
	cpu_nb = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
	cpu_nos = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
	cpu_son = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
	cpu_bc_exchange_buf = static_cast<Numeric4*>(memCPU.allocate(sizeof(Numeric4)*opts.grid_width));
	cudaHostRegister(cpu_bc_exchange_buf, sizeof(Numeric4)*opts.grid_width, cudaHostRegisterPortable);

	//Copy data from the input variable of type SWE_VAR_U to Numeric4 array
	#pragma omp parallel for
	for (int i = 1; i <= static_cast<int>(opts.grid_height); ++i)
	{
		for (int j = 1; j <= static_cast<int>(opts.grid_width); ++j)
		{
			cpu_system_state[i*(opts.grid_width + 2) + j].x = init_state.w[(i - 1)*opts.grid_width + j - 1];
			cpu_system_state[i*(opts.grid_width + 2) + j].y = init_state.hu[(i - 1)*opts.grid_width + j - 1];
			cpu_system_state[i*(opts.grid_width + 2) + j].z = init_state.hv[(i - 1)*opts.grid_width + j - 1];
		}
	}


	//***********************************Reconsider the following chunk of code...**********************************
	//Shared memory cost of the first two kernels, in bytes
	const unsigned int GPUShrdMemCost1 =
		(3 * SVS_CUDA_BLOCK_SIZE*SVS_CUDA_BLOCK_SIZE + SVS_CUDA_BLOCK_SIZE*(2 * SVS_CUDA_BLOCK_SIZE + 1))*sizeof(Numeric);
	//Shared memory cost of the second two kernels, in bytes
	const unsigned int GPUShrdMemCost2 =
		4 * SVS_CUDA_BLOCK_SIZE*SVS_CUDA_BLOCK_SIZE*sizeof(Numeric4);
	//Shared memory cost of the current subtask
	const unsigned int st_GPUShrdMemCost = GPUShrdMemCost1 > GPUShrdMemCost2 ? GPUShrdMemCost1 : GPUShrdMemCost2;
	//**************************************************************************************************************


	Numeric4 *st_system_state = cpu_system_state;	//pointer referring to the chunk of system state assigned to the i-th subtask
	const Numeric *st_B = opts.B;	//pointer referring to the chunk of topography component assigned to i-th task

	unsigned int num_grid_rows_distributed = 0;
	for (CudaDeviceScheduler::device_iterator cuda_device_iterator = CUDA_device_scheduler.begin();
		cuda_device_iterator != CUDA_device_scheduler.end();
		++cuda_device_iterator)
	{
		//Get numerical identifier of the current CUDA-capable device
		uint32_t id = cuda_device_iterator->getCudaDeviceId();

		//Compute number of grid rows assigned to the current device
		unsigned int num_ct_row_count = static_cast<unsigned int>(std::ceil(CUDA_device_scheduler.getLoadFactor(cuda_device_iterator)*opts.grid_height));
		num_ct_row_count = std::min<unsigned int>(opts.grid_height - num_grid_rows_distributed, num_ct_row_count);
		num_grid_rows_distributed += num_ct_row_count;
		st_grid_height[id] = num_ct_row_count;

		//Compute global memory cost of the subtask in bytes
		const unsigned int st_GPUGlblMemCost =
			sizeof(Numeric4)*(2 * (opts.grid_width + 2)*(st_grid_height[id] + 2) +    //system state and temporary buffer
			2 * (opts.grid_width + 1)*st_grid_height[id] +    //Te and Tw
			2 * opts.grid_width*(st_grid_height[id] + 1) +    //Tn and Ts
			2 * st_grid_height[id] +    //east-of-west and west-of-east boundary values
			2 * opts.grid_width +    //north-of-south and south-of-north boundary values
			2 * st_grid_height[id] +	//west and east boundary values
			opts.grid_width*st_grid_height[id] +	//rhs
			(id == best_device_id ? opts.grid_width : 0) +	//first CUDA-capable device allocates memory for the southern wall
			(id == worst_device_id ? opts.grid_width : 0)) +	//last CUDA-capable device allocates memory for the northern wall
			sizeof(Numeric)*(2 * opts.grid_width + 1)*(2 * st_grid_height[id] + 1);    //interpolated topography term


		//Check the amount of available global memory
		if (cuda_device_iterator->getTotalGlobalMemory() < st_GPUGlblMemCost)
		{
			error_state = true;
			error_descriptor = CudaErrorWrapper{ cudaErrorMemoryAllocation, __FILE__, __func__, __LINE__ };
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Check the amount of available shared memory per block
		if (cuda_device_iterator->getSharedMemoryPerBlock() < st_GPUShrdMemCost)
		{
			error_state = true;
			error_descriptor = CudaErrorWrapper{ cudaErrorMemoryAllocation, __FILE__, __func__, __LINE__ };
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Create the main CUDA command stream for the current device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaStreamCreate(&st_dev_stream[id]))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Create synchronization event for the current device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_ctxc[id]))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Create starting subtask event
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_start[id]))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Create ending subtask event
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventCreate(&st_finish[id]))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Initialize device memory storage
		memGPU.insert(std::make_pair(id, GPUMemoryBlock{}));
		memGPU[id].initialize(id, st_GPUGlblMemCost);

		//Allocate memory on the current device
		gpu_system_state[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
		gpu_temp[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2)));
		gpu_Te[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
		gpu_Tw[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 1)));
		gpu_Ts[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
		gpu_Tn[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*(st_grid_height[id] + 1)*opts.grid_width));
		gpu_rhs[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id] * opts.grid_width));
		gpu_eow[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
		gpu_woe[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
		gpu_nos[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
		gpu_son[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*opts.grid_width));
		gpu_wb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));
		gpu_eb[id] = static_cast<Numeric4*>(memGPU[id].allocate(sizeof(Numeric4)*st_grid_height[id]));

		if (id == best_device_id)	//the CUDA-capable device, which is enumerated the first, allocates memory for the southern wall
			gpu_sb = static_cast<Numeric4*>(memGPU[best_device_id].allocate(sizeof(Numeric4)*opts.grid_width));

		if (id == worst_device_id)	//the CUDA-capable device, which enumerates the last, allocates memory for the northern wall
			gpu_nb = static_cast<Numeric4*>(memGPU[worst_device_id].allocate(sizeof(Numeric4)*opts.grid_width));

		gpu_B[id] = static_cast<Numeric*>(memGPU[id].allocate(sizeof(Numeric)*(2 * st_grid_height[id] + 1)*(2 * opts.grid_width + 1)));

		//Copy initial state to the GPU memory
		cudaMemcpyAsync(gpu_system_state[id], st_system_state, sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2),
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		//Copy topography component to the GPU memory
		cudaMemcpyAsync(gpu_B[id], st_B, sizeof(Numeric)*(2 * opts.grid_width + 1)*(2 * st_grid_height[id] + 1),
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		if ((aux_err_desc = CudaErrorWrapper{ cudaPeekAtLastError(), __FILE__, __func__, __LINE__ }))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		st_system_state += st_grid_height[id] * (opts.grid_width + 2);
		st_B += 2 * st_grid_height[id] * (2 * opts.grid_width + 1);
	}

	is_initialized = true;	//switch object state to 'initialized' status
}


void SVSCore::initialize(const SVSVarU init_state, SVSParameters opts)
{
	initialize(init_state, opts.grid_height, opts.grid_width, opts.dx, opts.dy,
		opts.g, opts.theta, opts.eps, opts.B);
}


void SVSCore::loadBoundaryValues(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb)
{
	//Load western and eastern boundaries into the corresponding CPU buffers
	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(opts.grid_height); ++i)
	{
		//west
		cpu_wb[i].x = wb.w[i]; cpu_wb[i].y = wb.hu[i]; cpu_wb[i].z = wb.hv[i];

		//east-of-west
		cpu_eow[i].x = wb.w_edge[i]; cpu_eow[i].y = wb.hu_edge[i]; cpu_eow[i].z = wb.hv_edge[i];


		//east
		cpu_eb[i].x = eb.w[i]; cpu_eb[i].y = eb.hu[i]; cpu_eb[i].z = eb.hv[i];

		//west-of-east
		cpu_woe[i].x = eb.w_edge[i]; cpu_woe[i].y = eb.hu_edge[i]; cpu_woe[i].z = eb.hv_edge[i];
	}


	//Load southern and northern boundaries into the corresponding CPU buffers
	#pragma omp parallel for
	for (int j = 0; j < static_cast<int>(opts.grid_width); ++j)
	{
		//south
		cpu_sb[j].x = sb.w[j]; cpu_sb[j].y = sb.hu[j]; cpu_sb[j].z = sb.hv[j];

		//north-of-south
		cpu_nos[j].x = sb.w_edge[j]; cpu_nos[j].y = sb.hu_edge[j]; cpu_nos[j].z = sb.hv_edge[j];


		//north
		cpu_nb[j].x = nb.w[j]; cpu_nb[j].y = nb.hu[j]; cpu_nb[j].z = nb.hv[j];

		//south-of-north
		cpu_son[j].x = nb.w_edge[j]; cpu_son[j].y = nb.hu_edge[j]; cpu_son[j].z = nb.hv_edge[j];
	}


	//Copy boundary data to device memory
	Numeric4 *st_wb = cpu_wb, *st_eb = cpu_eb;
	Numeric4 *st_eow = cpu_eow, *st_woe = cpu_woe;
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the current device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		if (id == best_device_id)	//the device, which enumerates the first stores the southern and north-of-south boundaries
		{
			cudaMemcpyAsync(gpu_sb, cpu_sb, sizeof(Numeric4)*opts.grid_width,
				cudaMemcpyHostToDevice, st_dev_stream[best_device_id]);	//copy south boundary to GPU memory of the first device

			cudaMemcpyAsync(gpu_nos[best_device_id], cpu_nos, sizeof(Numeric4)*opts.grid_width,
				cudaMemcpyHostToDevice, st_dev_stream[best_device_id]);	//copy north-of-south boundary to GPU memory of the first device

			//setup auxiliary parameters for the kernel call to set south boundary values
			unsigned int _aux_grid_width = opts.grid_width % (4 * SVS_CUDA_BLOCK_SIZE) == 0 ? opts.grid_width :
				(opts.grid_width >> 2 + SVS_LOG_CUDA_BLOCK_SIZE << 2 + SVS_LOG_CUDA_BLOCK_SIZE) + 4 * SVS_CUDA_BLOCK_SIZE;
			dim3 __set_south_bc__grid_dim(_aux_grid_width / (4 * SVS_CUDA_BLOCK_SIZE));
			dim3 __set_south_bc__block_dim(4 * SVS_CUDA_BLOCK_SIZE);

			//set southern boundary values
			__set_south_bc__ <<<__set_south_bc__grid_dim, __set_south_bc__block_dim, 0, st_dev_stream[best_device_id] >>>
				(gpu_system_state[best_device_id], gpu_sb, opts.grid_width);
		}

		if (id == worst_device_id)	//the device, which enumerates the last stores northern and south-of-north boundaries
		{
			cudaMemcpyAsync(gpu_nb, cpu_nb, sizeof(Numeric4)*opts.grid_width,
				cudaMemcpyHostToDevice, st_dev_stream[worst_device_id]);	//copy north boundary to GPU memory of the last device

			cudaMemcpyAsync(gpu_son[worst_device_id], cpu_son, sizeof(Numeric4)*opts.grid_width,
				cudaMemcpyHostToDevice, st_dev_stream[worst_device_id]);	//copy south-of-north boundary to GPU memory of the last device

			//setup auxiliary parameters for the kernel call to set north boundary values
			unsigned int _aux_grid_width = opts.grid_width % (4 * SVS_CUDA_BLOCK_SIZE) == 0 ? opts.grid_width :
				(opts.grid_width >> 2 + SVS_LOG_CUDA_BLOCK_SIZE << 2 + SVS_LOG_CUDA_BLOCK_SIZE) + 4 * SVS_CUDA_BLOCK_SIZE;
			dim3 __set_north_bc__grid_dim(_aux_grid_width / (4 * SVS_CUDA_BLOCK_SIZE));
			dim3 __set_north_bc__block_dim(4 * SVS_CUDA_BLOCK_SIZE);

			//set northern boundary values
			__set_north_bc__ <<<__set_north_bc__grid_dim, __set_north_bc__block_dim, 0, st_dev_stream[worst_device_id] >>>
				(gpu_system_state[worst_device_id], gpu_nb, st_grid_height[worst_device_id], opts.grid_width);
		}

		//copy western boundary
		cudaMemcpyAsync(gpu_wb[id], st_wb, sizeof(Numeric4)*st_grid_height[id],
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		//copy eastern boundary
		cudaMemcpyAsync(gpu_eb[id], st_eb, sizeof(Numeric4)*st_grid_height[id],
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		//copy east-of-west boundary
		cudaMemcpyAsync(gpu_eow[id], st_eow, sizeof(Numeric4)*st_grid_height[id],
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		//copy west-of-east boundary
		cudaMemcpyAsync(gpu_woe[id], st_woe, sizeof(Numeric4)*st_grid_height[id],
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		//setup auxiliary parameters for the kernel call to set west and east boundary values
		unsigned int _aux_grid_height = st_grid_height[id] % (4 * SVS_CUDA_BLOCK_SIZE) == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> 2 + SVS_LOG_CUDA_BLOCK_SIZE << 2 + SVS_LOG_CUDA_BLOCK_SIZE) + 4 * SVS_CUDA_BLOCK_SIZE;
		dim3 __set_west_east_bc__grid_dim(_aux_grid_height / (4 * SVS_CUDA_BLOCK_SIZE));
		dim3 __set_west_east_bc__block_dim(4 * SVS_CUDA_BLOCK_SIZE);

		//set west and east boundary values for the current task
		__set_west_east_bc__ <<<__set_west_east_bc__grid_dim, __set_west_east_bc__block_dim, 0, st_dev_stream[id] >>>
			(gpu_system_state[id], gpu_wb[id], gpu_eb[id], st_grid_height[id], opts.grid_width);

		//update pointers
		st_wb += st_grid_height[id];
		st_eb += st_grid_height[id];
		st_eow += st_grid_height[id];
		st_woe += st_grid_height[id];
	}
}


void SVSCore::computeRHS()
{
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve identifier of the current device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Execute the first kernel to compute cardinal direction values
		unsigned int _aux_grid_width = opts.grid_width % SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __compute_cardinals__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __compute_cardinals__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__compute_cardinals__ <<<__compute_cardinals__grid_dim, __compute_cardinals__block_dim, 0, st_dev_stream[id] >>>
			(gpu_Te[id], gpu_Tw[id], gpu_Tn[id], gpu_Ts[id], gpu_system_state[id] + opts.grid_width + 3, gpu_B[id], opts.dx, opts.dy,
			opts.grid_width, st_grid_height[id], opts.theta, opts.eps);

		//Execute the second kernel to set eastern and western boundaries for the cardinals
		_aux_grid_height = st_grid_height[id] % (4 * SVS_CUDA_BLOCK_SIZE) == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> 2 + SVS_LOG_CUDA_BLOCK_SIZE << 2 + SVS_LOG_CUDA_BLOCK_SIZE) + 4 * SVS_CUDA_BLOCK_SIZE;
		dim3 __set_eow_woe_bc__grid_dim(_aux_grid_height / (4 * SVS_CUDA_BLOCK_SIZE));
		dim3 __set_eow_woe_bc__block_dim(4 * SVS_CUDA_BLOCK_SIZE);
		__set_eow_woe_bc__ <<<__set_eow_woe_bc__grid_dim, __set_eow_woe_bc__block_dim, 0, st_dev_stream[id] >>>
			(gpu_Te[id], gpu_Tw[id], gpu_eow[id], gpu_woe[id], gpu_B[id], opts.eps, opts.grid_width, st_grid_height[id]);
	}


	for (CudaDeviceScheduler::device_iterator cuda_device_iterator = CUDA_device_scheduler.begin();
		cuda_device_iterator != CUDA_device_scheduler.end();
		++cuda_device_iterator)
	{
		//Retrieve identifier of the current device
		uint32_t id = cuda_device_iterator->getCudaDeviceId();
		
		bool reconstructed_nos = false;
		bool reconstructed_son = false;

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}


		//Compute RHS of the related ODE system

		//Calculate number of blocks for the 3rd kernel
		unsigned int _aux_grid_width = opts.grid_width % (4 * SVS_CUDA_BLOCK_SIZE) == 0 ? opts.grid_width :
			(opts.grid_width >> 2 + SVS_LOG_CUDA_BLOCK_SIZE << 2 + SVS_LOG_CUDA_BLOCK_SIZE) + 4 * SVS_CUDA_BLOCK_SIZE;
		dim3 __set_nos_son_bc__grid_dim(_aux_grid_width / (4 * SVS_CUDA_BLOCK_SIZE), 1, 1);
		dim3 __set_nos_son_bc__block_dim(4 * SVS_CUDA_BLOCK_SIZE, 1, 1);


		//Account for north-of-south and south-of-north boundary values

		//North-Of-South
		if (id != best_device_id)
		{
			uint32_t previous_device_id = (--cuda_device_iterator)->getCudaDeviceId(); ++cuda_device_iterator;

			//Make active the device, which is enumerated "right before" the current
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(previous_device_id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = previous_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}
			cudaMemcpyAsync(cpu_bc_exchange_buf, gpu_Tn[previous_device_id] + st_grid_height[previous_device_id] * opts.grid_width,
				sizeof(Numeric4)*opts.grid_width, cudaMemcpyDeviceToHost, st_dev_stream[previous_device_id]);

			//Record synchronization event
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventRecord(st_ctxc[previous_device_id], st_dev_stream[previous_device_id]))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = previous_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Activate the device, which has been active on this iteration of the sub-task assignment loop
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Synchronize "this" device with the "previous" device
			cudaStreamWaitEvent(st_dev_stream[id], st_ctxc[previous_device_id], 0);
			cudaMemcpyAsync(gpu_nos[id], cpu_bc_exchange_buf, sizeof(Numeric4)*opts.grid_width, cudaMemcpyHostToDevice, st_dev_stream[id]);
			
			reconstructed_nos = true;
		}

		//South-Of-North
		if (id != worst_device_id)
		{
			uint32_t next_device_id = (++cuda_device_iterator)->getCudaDeviceId(); --cuda_device_iterator;

			//Make active the device, which is enumerated "after" the current
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(next_device_id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = next_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}
			cudaMemcpyAsync(cpu_bc_exchange_buf, gpu_Ts[next_device_id], sizeof(Numeric4)*opts.grid_width,
				cudaMemcpyDeviceToHost, st_dev_stream[next_device_id]);

			//Record synchronization event
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventRecord(st_ctxc[next_device_id], st_dev_stream[next_device_id]))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = next_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Activate the device, which has been previously active on this iteration of the sub-task distribution loop
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Synchronize "this" device with the "next" device
			cudaStreamWaitEvent(st_dev_stream[id], st_ctxc[next_device_id], 0);
			cudaMemcpyAsync(gpu_son[id], cpu_bc_exchange_buf, sizeof(Numeric4)*opts.grid_width, cudaMemcpyHostToDevice, st_dev_stream[id]);

			reconstructed_son = true;
		}

		//Launch the 3rd kernel
		__set_nos_son_bc__<<<__set_nos_son_bc__grid_dim, __set_nos_son_bc__block_dim, 0, st_dev_stream[id]>>>
			(gpu_Tn[id], gpu_Ts[id], gpu_nos[id], gpu_son[id], gpu_B[id], opts.eps, opts.grid_width, st_grid_height[id], reconstructed_nos, reconstructed_son);

		//Check if there were any errors
		if ((aux_err_desc = CudaErrorWrapper{ cudaPeekAtLastError(), __FILE__, __func__, __LINE__ }))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Execute the fourth kernel and get the part of RHS related to the corresponding subtask
		_aux_grid_width = opts.grid_width % SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __compute_rhs__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __compute_rhs__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__compute_rhs__<<<__compute_rhs__grid_dim, __compute_rhs__block_dim, 0, st_dev_stream[id]>>>
			(gpu_rhs[id], gpu_Te[id], gpu_Tw[id], gpu_Tn[id], gpu_Ts[id], gpu_system_state[id] + opts.grid_width + 3, gpu_B[id],
			opts.dx, opts.dy, opts.g, opts.grid_width, st_grid_height[id]);
	}
}


void SVSCore::cloneSystemState()
{
	for (auto device : CUDA_device_scheduler)
	{
		uint32_t id = device.getCudaDeviceId();
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}
		cudaMemcpyAsync(gpu_temp[id], gpu_system_state[id], sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2),
			cudaMemcpyDeviceToDevice, st_dev_stream[id]);
	}
}


void SVSCore::reconstructBoundaries()
{
	//Perform correction of the boundary values across the devices
	//Consider replacing cudaMemcpy(Async)(...) calls below with kernel launches.
	//The present implementation is a workaround hack

	for (CudaDeviceScheduler::device_iterator cuda_device_iterator = CUDA_device_scheduler.begin();
		cuda_device_iterator != CUDA_device_scheduler.end();
		++cuda_device_iterator)
	{
		uint32_t id = cuda_device_iterator->getCudaDeviceId();	//retrieve numerical identifier of the current device

		if (id != best_device_id)	//reconstruction of the southern boundary is needed
		{
			uint32_t previous_device_id = (--cuda_device_iterator)->getCudaDeviceId(); ++cuda_device_iterator;

			//Switch to the device, which "precedes" the currently enumerated device
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(previous_device_id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = previous_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Copy the last row of system state's chunk of the previous device to the CPU boundary value exchange buffer
			cudaMemcpyAsync(cpu_bc_exchange_buf, gpu_system_state[previous_device_id] + st_grid_height[previous_device_id] * (opts.grid_width + 2) + 1,
				sizeof(Numeric4)*opts.grid_width, cudaMemcpyDeviceToHost, st_dev_stream[previous_device_id]);

			//Record synchronization event
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaEventRecord(st_ctxc[previous_device_id], st_dev_stream[previous_device_id]))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = previous_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Activate device enumerated on this iteration of the device enumeration loop
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Synchronize "this" device with the "previous" device
			cudaStreamWaitEvent(st_dev_stream[id], st_ctxc[previous_device_id], 0);

			//Copy the southern boundary to the current device
			cudaMemcpyAsync(gpu_system_state[id] + 1, cpu_bc_exchange_buf,
				sizeof(Numeric4)*opts.grid_width, cudaMemcpyHostToDevice, st_dev_stream[id]);
		}

		if (id != worst_device_id)	//reconstruction of the northern boundary is needed
		{
			//Retrieve numerical identifier of the device, which is enumerated after the current device
			uint32_t next_device_id = (++cuda_device_iterator)->getCudaDeviceId(); --cuda_device_iterator;

			//Activate the "next" device
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(next_device_id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = next_device_id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Copy the first row of system state's chunk of the next device to the CPU boundary value exchange buffer
			cudaMemcpyAsync(cpu_bc_exchange_buf, gpu_system_state[next_device_id] + (opts.grid_width + 2) + 1,
				sizeof(Numeric4)*opts.grid_width, cudaMemcpyDeviceToHost, st_dev_stream[next_device_id]);

			//Record synchronization event
			cudaEventRecord(st_ctxc[next_device_id], st_dev_stream[next_device_id]);

			//Switch back to the "current" device
			if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
			{
				error_state = true;
				error_descriptor = aux_err_desc;
				error_source = id;
				error_callback(error_descriptor, error_source);
				return;
			}

			//Synchronize "this" device with the "next" device
			cudaStreamWaitEvent(st_dev_stream[id], st_ctxc[next_device_id], 0);

			//Copy the northern boundary to the current device
			cudaMemcpyAsync(gpu_system_state[id] + (st_grid_height[id] + 1)*(opts.grid_width + 2) + 1, cpu_bc_exchange_buf,
				sizeof(Numeric4)*opts.grid_width, cudaMemcpyHostToDevice, st_dev_stream[id]);
		}
	}
}


void SVSCore::solveEuler(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt)
{
	//Check if the object is not in an erroneous state, and that it has already been initialized
	if (error_state || !is_initialized) return;

	//Load supplied boundary conditions into internal buffers
	loadBoundaryValues(wb, eb, sb, nb);

	//Compute RHS(U0)
	computeRHS();

	//Make an Euler step
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch the Euler step kernel
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __euler__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __euler__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__euler__ <<<__euler__grid_dim, __euler__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}

	//Reconstruct boundary values
	reconstructBoundaries();
}


void SVSCore::solveSSPRK22(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt)
{
	//Check if the object is not in an erroneous state, and that it has already been initialized
	if (error_state || !is_initialized) return;

	//Load supplied boundary conditions into internal buffers
	loadBoundaryValues(wb, eb, sb, nb);

	//Make temporary copy of the current state
	cloneSystemState();

	//Compute RHS(U0)
	computeRHS();

	//Make an Euler step
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch the Euler step kernel
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __euler__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __euler__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__euler__<<<__euler__grid_dim, __euler__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}


	//Reconstruct boundary values
	reconstructBoundaries();

	//Compute RHS(U1)
	computeRHS();


	//Make the second semi-step of SSP-RK(2,2)
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch the Euler step kernel
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __ssprk22_2__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __ssprk22_2__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__ssprk22_2__<<<__ssprk22_2__grid_dim, __ssprk22_2__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_temp[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}

	//Reconstruct boundary values
	reconstructBoundaries();
}


void SVSCore::solveSSPRK33(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt)
{
	//Check if the object is not in an erroneous state, and that it has already been initialized
	if (error_state || !is_initialized) return;

	//Load supplied boundary conditions into internal buffers
	loadBoundaryValues(wb, eb, sb, nb);

	//Make temporary copy of the current state
	cloneSystemState();

	//Compute RHS(U0)
	computeRHS();

	//Make an Euler step
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch the Euler step kernel
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __euler__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __euler__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__euler__<<<__euler__grid_dim, __euler__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}


	//Reconstruct boundary values
	reconstructBoundaries();

	//Compute RHS(U1)
	computeRHS();


	//Make the second semi-step of SSP-RK(3,3)
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch CUDA-kernel implementing the second semi-step of SSP-RK(3,3)
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __ssprk33_2__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __ssprk33_2__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__ssprk33_2__<<<__ssprk33_2__grid_dim, __ssprk33_2__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_temp[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}


	//Reconstruct boundary values
	reconstructBoundaries();

	//Compute RHS(U2)
	computeRHS();


	//Make the last semi-step of SSP-RK(3,3)
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Setting up auxiliary parameters to launch CUDA-kernel implementing the last semi-step of SSP-RK(3,3)
		unsigned int _aux_grid_width = opts.grid_width%SVS_CUDA_BLOCK_SIZE == 0 ? opts.grid_width :
			(opts.grid_width >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		unsigned int _aux_grid_height = st_grid_height[id] % SVS_CUDA_BLOCK_SIZE == 0 ? st_grid_height[id] :
			(st_grid_height[id] >> SVS_LOG_CUDA_BLOCK_SIZE << SVS_LOG_CUDA_BLOCK_SIZE) + SVS_CUDA_BLOCK_SIZE;
		dim3 __ssprk33_3__grid_dim(_aux_grid_width / SVS_CUDA_BLOCK_SIZE, _aux_grid_height / SVS_CUDA_BLOCK_SIZE);
		dim3 __ssprk33_3__block_dim(SVS_CUDA_BLOCK_SIZE, SVS_CUDA_BLOCK_SIZE);
		__ssprk33_3__<<<__ssprk33_3__grid_dim, __ssprk33_3__block_dim, 0, st_dev_stream[id]>>>
			(gpu_system_state[id], gpu_temp[id], gpu_rhs[id], dt, st_grid_height[id], opts.grid_width);
	}


	//Reconstruct boundaries
	reconstructBoundaries();
}


const Numeric4 *SVSCore::getSystemState() const
{
	Numeric4 *st_cpu_system_state = cpu_system_state;
	uint32_t best_device_id = CUDA_device_scheduler.bestDevice()->getCudaDeviceId();

	//Copy data from the first CUDA-capable device
	if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(best_device_id))))
	{
		error_state = true;
		error_descriptor = aux_err_desc;
		error_source = best_device_id;
		error_callback(error_descriptor, error_source);
		return nullptr;
	}
	cudaMemcpy(st_cpu_system_state, gpu_system_state[best_device_id],
		sizeof(Numeric4)*(st_grid_height[best_device_id] + 2)*(opts.grid_width + 2), cudaMemcpyDeviceToHost);
	st_cpu_system_state += (st_grid_height[best_device_id] + 2)*(opts.grid_width + 2);

	//Copy data from the rest of CUDA-capable devices to CPU memory
	for (CudaDeviceScheduler::device_iterator cuda_device_iterator = ++CUDA_device_scheduler.begin();
		cuda_device_iterator != CUDA_device_scheduler.end();
		++cuda_device_iterator)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = cuda_device_iterator->getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return nullptr;
		}

		cudaMemcpy(st_cpu_system_state, gpu_system_state[id] + 2 * (opts.grid_width + 2),
			sizeof(Numeric4)*st_grid_height[id] * (opts.grid_width + 2), cudaMemcpyDeviceToHost);
		st_cpu_system_state += st_grid_height[id] * (opts.grid_width + 2);
	}
	return cpu_system_state;
}


void SVSCore::setGravityConstant(Numeric g) { opts.g = g; }

Numeric SVSCore::getGravityConstant() const { return opts.g; }


void SVSCore::setTheta(Numeric theta) { opts.theta = theta; }

Numeric SVSCore::getTheta() const { return opts.theta; }


void SVSCore::setEpsilon(Numeric epsilon) { opts.eps = epsilon; }

Numeric SVSCore::getEpsilon() const { return opts.eps; }

SVSCore::DomainSettings SVSCore::getDomainSettings() const { return DomainSettings{ opts.grid_width, opts.grid_height, opts.dx, opts.dy }; }


void SVSCore::updateSystemState(SVSVarU new_system_state)
{
	//Convert user-supplied system state to the internal representation
	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(opts.grid_height); ++i)
	{
		for (int j = 0; j < static_cast<int>(opts.grid_width); ++j)
		{
			cpu_system_state[(i + 1)*(opts.grid_width + 2) + j + 1].x = new_system_state.w[i*opts.grid_width + j];
			cpu_system_state[(i + 1)*(opts.grid_width + 2) + j + 1].y = new_system_state.hu[i*opts.grid_width + j];
			cpu_system_state[(i + 1)*(opts.grid_width + 2) + j + 1].z = new_system_state.hv[i*opts.grid_width + j];
		}
	}

	//Copy updated system state to GPU memory
	Numeric4 *st_cpu_system_state = cpu_system_state;
	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		cudaMemcpyAsync(gpu_system_state[id], st_cpu_system_state,
			sizeof(Numeric4)*(st_grid_height[id] + 2)*(opts.grid_width + 2), cudaMemcpyHostToDevice, st_dev_stream[id]);
		st_cpu_system_state += st_grid_height[id] * (opts.grid_width + 2);
	}
}


void SVSCore::updateTopographyData(const Numeric* B)
{
	opts.B = B;	//update pointer for topography data
	const Numeric *st_B = opts.B;	//pointer referring to the chunk of topography component assigned to i-th task

	for (auto device : CUDA_device_scheduler)
	{
		//Retrieve numerical identifier of the currently enumerated device
		uint32_t id = device.getCudaDeviceId();

		//Switch active device
		if ((aux_err_desc = CUDA_SAFE_CALL_NO_RETURN(cudaSetDevice(id))))
		{
			error_state = true;
			error_descriptor = aux_err_desc;
			error_source = id;
			error_callback(error_descriptor, error_source);
			return;
		}

		//Copy topography component to the GPU memory
		cudaMemcpyAsync(gpu_B[id], st_B, sizeof(Numeric)*(2 * opts.grid_width + 1)*(2 * st_grid_height[id] + 1),
			cudaMemcpyHostToDevice, st_dev_stream[id]);

		st_B += 2 * st_grid_height[id] * (2 * opts.grid_width + 1);
	}
}




//*********************************************************GPGPU functions implementations*********************************************************
template<typename block_type>
__device__ const block_type __get_block_element__(const block_type *grid, int block_x, int block_y, int x, int y, int stride)
{
	return grid[(SVS_CUDA_BLOCK_SIZE*block_y + y)*stride + (SVS_CUDA_BLOCK_SIZE*block_x + x)];
}


template<typename block_type>
__device__ void __set_block_element__(block_type *grid, int block_x, int block_y, int x, int y, int stride, block_type element)
{
	grid[(SVS_CUDA_BLOCK_SIZE*block_y + y)*stride + (SVS_CUDA_BLOCK_SIZE*block_x + x)] = element;
}


__device__ const Numeric* __get_topography_block_element_addr__(const Numeric* B, int block_x, int block_y, int x, int y, int grid_width)
{
	return (B + (1 + (SVS_CUDA_BLOCK_SIZE*block_y + y) * 2)*(2 * grid_width + 1) + (1 + (SVS_CUDA_BLOCK_SIZE*block_x + x) * 2));
}


__device__ Numeric __minmod__(Numeric a1, Numeric a2, Numeric a3)
{
	bool _case1 = a1 > 0 && a2 > 0 && a3 > 0;
	bool _case2 = a1 < 0 && a2 < 0 && a3 < 0;
	return _case1*_min(_min(a1, a2), a3) + _case2*_max(_max(a1, a2), a3);
}


__device__ __inline__ Numeric3 __flux_F__(Numeric4 T, Numeric B, Numeric g)
{
	Numeric3 F = { 0, 0, 0 };
	Numeric _h = T.w - B;

	//Account for the dry areas
	if (_h == 0) return F;

	F.x = T.y;    //(hu)
	F.y = (T.y*T.y) / _h + g / 2 * (_h*_h);    //((hu)^2/(w-B)+g/2*(w-B)^2
	F.z = (T.y*T.z) / _h;    //((hu)(hv)/(w-B)
	return F;
}


__device__ __inline__ Numeric3 __flux_G__(Numeric4 T, Numeric B, Numeric g)
{
	Numeric3 G = { 0, 0, 0 };
	Numeric _h = T.w - B;

	//Account for the dry areas
	if (_h == 0) return G;

	G.x = T.z;    //(hv)
	G.y = (T.y*T.z) / _h;    //((hu)(hv)/(w-B)
	G.z = (T.z*T.z) / _h + g / 2 * (_h*_h);    //(hv)^2/(w-B)+g/2*(w-B)^2
	return G;
}


__device__ __inline__ Numeric3 __flux_Hx__(Numeric4 _Te, Numeric4 _Tw, Numeric g)
{
	Numeric3 Hx = { 0, 0, 0 };    //numeric horizontal flux Hx

	//Correct negative water heights
	_Te.x = _max(_Te.x, 0);
	_Tw.x = _max(_Tw.x, 0);

	//Compute horizontal propagation speeds
	Numeric _aux1 = _sqrt(g*_Te.x);
	Numeric _aux2 = _sqrt(g*_Tw.x);
	Numeric ap = _max(_max(_Te.y + _aux1, _Tw.y + _aux2), 0);
	Numeric am = _min(_min(_Te.y - _aux1, _Tw.y - _aux2), 0);

	//Account for the regions with no horizontal flow
	if (ap == am)
		return Hx;

	//update discharges
	_Te.y *= _Te.x; _Te.z *= _Te.x;
	_Tw.y *= _Tw.x; _Tw.z *= _Tw.x;

	//compute horizontal flux F
	Numeric Be = _Te.w - _Te.x;    //B(j+1/2,k)
	Numeric3 Fe = __flux_F__(_Te, Be, g);
	Fe.x *= ap; Fe.y *= ap; Fe.z *= ap;
	Numeric3 Fw = __flux_F__(_Tw, Be, g);
	Fw.x *= am; Fw.y *= am; Fw.z *= am;

	Numeric _sp_ratio = (ap*am) / (ap - am);
	Hx.x = (Fe.x - Fw.x) / (ap - am) + _sp_ratio*(_Tw.w - _Te.w);
	Hx.y = (Fe.y - Fw.y) / (ap - am) + _sp_ratio*(_Tw.y - _Te.y);
	Hx.z = (Fe.z - Fw.z) / (ap - am) + _sp_ratio*(_Tw.z - _Te.z);

	return Hx;
}


__device__ __inline__ Numeric3 __flux_Hy__(Numeric4 _Tn, Numeric4 _Ts, Numeric g)
{
	Numeric3 Hy = { 0, 0, 0 };    //numeric vertical flux Hy

	//Correct negative water heights
	_Tn.x = _max(_Tn.x, 0);
	_Ts.x = _max(_Ts.x, 0);

	//Compute vertical propagation speeds
	Numeric _aux1 = _sqrt(g*_Tn.x);
	Numeric _aux2 = _sqrt(g*_Ts.x);
	Numeric bp = _max(_max(_Tn.z + _aux1, _Ts.z + _aux2), 0);
	Numeric bm = _min(_min(_Tn.z - _aux1, _Ts.z - _aux2), 0);

	//Account for the regions with no vertical flow
	if (bp == bm)
		return Hy;

	//update discharges
	_Tn.y *= _Tn.x; _Tn.z *= _Tn.x;
	_Ts.y *= _Ts.x; _Ts.z *= _Ts.x;

	//compute vertical flux G
	Numeric Bn = _Tn.w - _Tn.x;    //B(j,k+1/2)
	Numeric3 Gn = __flux_G__(_Tn, Bn, g);
	Gn.x *= bp; Gn.y *= bp; Gn.z *= bp;
	Numeric3 Gs = __flux_G__(_Ts, Bn, g);
	Gs.x *= bm; Gs.y *= bm; Gs.z *= bm;

	Numeric _sp_ratio = (bp*bm) / (bp - bm);
	Hy.x = (Gn.x - Gs.x) / (bp - bm) + _sp_ratio*(_Ts.w - _Tn.w);
	Hy.y = (Gn.y - Gs.y) / (bp - bm) + _sp_ratio*(_Ts.y - _Tn.y);
	Hy.z = (Gn.z - Gs.z) / (bp - bm) + _sp_ratio*(_Ts.z - _Tn.z);
	return Hy;
}


//CUDA Kernel, which computes U-values at the cardinal directions of each finite-volume cell
__global__ void __compute_cardinals__(Numeric4* Te, Numeric4* Tw, Numeric4* Tn, Numeric4* Ts, const Numeric4* U, const Numeric* B,
	Numeric dx, Numeric dy, unsigned int grid_width, unsigned int grid_height, Numeric theta, Numeric eps)
	//Block U already contains boundary conditions, but supplied pointer must point to the upper-left corner of the "internal" area
	//Here T=(h,u,v,w) (cardinal direction literals are intentionally omitted)
	//Te, Tw, Tn and Ts allocate extra space for boundary conditions in accordance with the following convention:
	//Te: first column; Tw: last column; Tn: first row; Ts: last row
{
	//get block address
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	//get thread address
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	//get absolute indexes of the thread
	int idx_x = block_x*SVS_CUDA_BLOCK_SIZE + thread_x;
	int idx_y = block_y*SVS_CUDA_BLOCK_SIZE + thread_y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;


	//Each thread reads the corresponding block element to the shared memory.
	__shared__ Numeric w[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric hu[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric hv[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric topography[SVS_CUDA_BLOCK_SIZE][2 * SVS_CUDA_BLOCK_SIZE + 1];

	//Read the corresponding sub-task grid node to the shared memory
	w[thread_y][thread_x] = __get_block_element__(U, block_x, block_y, thread_x, thread_y, grid_width + 2).x;
	hu[thread_y][thread_x] = __get_block_element__(U, block_x, block_y, thread_x, thread_y, grid_width + 2).y;
	hv[thread_y][thread_x] = __get_block_element__(U, block_x, block_y, thread_x, thread_y, grid_width + 2).z;

	//Read topography elements

	//west component
	topography[thread_y][2 * thread_x] =
		*(__get_topography_block_element_addr__(B, block_x, block_y, thread_x, thread_y, grid_width) - 1);

	//north component
	topography[thread_y][2 * thread_x + 1] =
		*(__get_topography_block_element_addr__(B, block_x, block_y, thread_x, thread_y, grid_width) + (2 * grid_width + 1));

	//account for the eastern boundary
	if (thread_x == SVS_CUDA_BLOCK_SIZE - 1 || idx_x == grid_width - 1)
		topography[thread_y][2 * thread_x + 2] =
		*(__get_topography_block_element_addr__(B, block_x, block_y, thread_x, thread_y, grid_width) + 1);

	//synchronize threads
	__syncthreads();



	//Compute numerical derivatives in zonal direction
	Numeric wdx; Numeric hudx; Numeric hvdx;

	//Account for the west boundary
	if (thread_x == 0)
	{
		wdx = __minmod__(theta / dx*(w[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).x),
			(w[thread_y][thread_x + 1] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).x) / (2 * dx),
			theta / dx*(w[thread_y][thread_x + 1] - w[thread_y][thread_x]));
		hudx = __minmod__(theta / dx*(hu[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).y),
			(hu[thread_y][thread_x + 1] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).y) / (2 * dx),
			theta / dx*(hu[thread_y][thread_x + 1] - hu[thread_y][thread_x]));
		hvdx = __minmod__(theta / dx*(hv[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).z),
			(hv[thread_y][thread_x + 1] - __get_block_element__(U, block_x, block_y, -1, thread_y, grid_width + 2).z) / (2 * dx),
			theta / dx*(hv[thread_y][thread_x + 1] - hv[thread_y][thread_x]));
	}

	//Account for east boundary
	if (thread_x == SVS_CUDA_BLOCK_SIZE - 1 || idx_x == grid_width - 1)
	{
		wdx = __minmod__(theta / dx*(w[thread_y][thread_x] - w[thread_y][thread_x - 1]),
			(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).x - w[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).x - w[thread_y][thread_x]));
		hudx = __minmod__(theta / dx*(hu[thread_y][thread_x] - hu[thread_y][thread_x - 1]),
			(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).y - hu[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).y - hu[thread_y][thread_x]));
		hvdx = __minmod__(theta / dx*(hv[thread_y][thread_x] - hv[thread_y][thread_x - 1]),
			(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).z - hv[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(__get_block_element__(U, block_x, block_y, thread_x + 1, thread_y, grid_width + 2).z - hv[thread_y][thread_x]));
	}

	//Rest of the zonal derivatives
	if (thread_x > 0 && thread_x < SVS_CUDA_BLOCK_SIZE - 1 && idx_x < grid_width - 1)
	{
		wdx = __minmod__(theta / dx*(w[thread_y][thread_x] - w[thread_y][thread_x - 1]),
			(w[thread_y][thread_x + 1] - w[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(w[thread_y][thread_x + 1] - w[thread_y][thread_x]));
		hudx = __minmod__(theta / dx*(hu[thread_y][thread_x] - hu[thread_y][thread_x - 1]),
			(hu[thread_y][thread_x + 1] - hu[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(hu[thread_y][thread_x + 1] - hu[thread_y][thread_x]));
		hvdx = __minmod__(theta / dx*(hv[thread_y][thread_x] - hv[thread_y][thread_x - 1]),
			(hv[thread_y][thread_x + 1] - hv[thread_y][thread_x - 1]) / (2 * dx),
			theta / dx*(hv[thread_y][thread_x + 1] - hv[thread_y][thread_x]));
	}


	//Compute numerical derivatives in meridional direction
	Numeric wdy; Numeric hudy; Numeric hvdy;

	//Account for the south boundary
	if (thread_y == 0)
	{
		wdy = __minmod__(theta / dy*(w[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).x),
			(w[thread_y + 1][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).x) / (2 * dy),
			theta / dy*(w[thread_y + 1][thread_x] - w[thread_y][thread_x]));
		hudy = __minmod__(theta / dy*(hu[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).y),
			(hu[thread_y + 1][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).y) / (2 * dy),
			theta / dy*(hu[thread_y + 1][thread_x] - hu[thread_y][thread_x]));
		hvdy = __minmod__(theta / dy*(hv[thread_y][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).z),
			(hv[thread_y + 1][thread_x] - __get_block_element__(U, block_x, block_y, thread_x, -1, grid_width + 2).z) / (2 * dy),
			theta / dy*(hv[thread_y + 1][thread_x] - hv[thread_y][thread_x]));
	}

	//Account for north boundary
	if (thread_y == SVS_CUDA_BLOCK_SIZE - 1 || idx_y == grid_height - 1)
	{
		wdy = __minmod__(theta / dy*(w[thread_y][thread_x] - w[thread_y - 1][thread_x]),
			(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).x - w[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).x - w[thread_y][thread_x]));
		hudy = __minmod__(theta / dy*(hu[thread_y][thread_x] - hu[thread_y - 1][thread_x]),
			(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).y - hu[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).y - hu[thread_y][thread_x]));
		hvdy = __minmod__(theta / dy*(hv[thread_y][thread_x] - hv[thread_y - 1][thread_x]),
			(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).z - hv[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(__get_block_element__(U, block_x, block_y, thread_x, thread_y + 1, grid_width + 2).z - hv[thread_y][thread_x]));
	}

	//Rest of the meridional derivatives
	if (thread_y > 0 && thread_y < SVS_CUDA_BLOCK_SIZE - 1 && idx_y < grid_height - 1)
	{
		wdy = __minmod__(theta / dy*(w[thread_y][thread_x] - w[thread_y - 1][thread_x]),
			(w[thread_y + 1][thread_x] - w[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(w[thread_y + 1][thread_x] - w[thread_y][thread_x]));
		hudy = __minmod__(theta / dy*(hu[thread_y][thread_x] - hu[thread_y - 1][thread_x]),
			(hu[thread_y + 1][thread_x] - hu[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(hu[thread_y + 1][thread_x] - hu[thread_y][thread_x]));
		hvdy = __minmod__(theta / dy*(hv[thread_y][thread_x] - hv[thread_y - 1][thread_x]),
			(hv[thread_y + 1][thread_x] - hv[thread_y - 1][thread_x]) / (2 * dy),
			theta / dy*(hv[thread_y + 1][thread_x] - hv[thread_y][thread_x]));
	}



	//Compute cardinal values
	//East
	Numeric we = w[thread_y][thread_x] + dx / 2 * wdx;
	Numeric hue = hu[thread_y][thread_x] + dx / 2 * hudx;
	Numeric hve = hv[thread_y][thread_x] + dx / 2 * hvdx;

	//West
	Numeric ww = w[thread_y][thread_x] - dx / 2 * wdx;
	Numeric huw = hu[thread_y][thread_x] - dx / 2 * hudx;
	Numeric hvw = hv[thread_y][thread_x] - dx / 2 * hvdx;

	//North
	Numeric wn = w[thread_y][thread_x] + dy / 2 * wdy;
	Numeric hun = hu[thread_y][thread_x] + dy / 2 * hudy;
	Numeric hvn = hv[thread_y][thread_x] + dy / 2 * hvdy;

	//South
	Numeric ws = w[thread_y][thread_x] - dy / 2 * wdy;
	Numeric hus = hu[thread_y][thread_x] - dy / 2 * hudy;
	Numeric hvs = hv[thread_y][thread_x] - dy / 2 * hvdy;



	//Correct cardinal values

	//Get topography values
	Numeric Be, Bw, Bn, Bs;
	Bw = topography[thread_y][2 * thread_x];
	Bn = topography[thread_y][2 * thread_x + 1];
	Be = topography[thread_y][2 * thread_x + 2];
	if (thread_y == 0)
		Bs = *(__get_topography_block_element_addr__(B, block_x, block_y, thread_x, 0, grid_width) - (2 * grid_width + 1));
	else
		Bs = topography[thread_y - 1][2 * thread_x + 1];

	//Correct water levels
	if (we < Be)
	{
		we = Be;
		ww = 2 * w[thread_y][thread_x] - Be;
	}
	if (ww < Bw)
	{
		we = 2 * w[thread_y][thread_x] - Bw;
		ww = Bw;
	}
	if (wn < Bn)
	{
		wn = Bn;
		ws = 2 * w[thread_y][thread_x] - Bn;
	}
	if (ws < Bs)
	{
		wn = 2 * w[thread_y][thread_x] - Bs;
		ws = Bs;
	}

	//Get water elevations
	Numeric he = we - Be, hw = ww - Bw, hn = wn - Bn, hs = ws - Bs;


	//Calculate speed field by de-singularizing the discharges

	//East
	Numeric _aux = he*he*he*he;
	_aux = _sqrt(2)*he / _sqrt(_aux + _max(_aux, eps));
	Numeric ue = _aux*hue, ve = _aux*hve;

	//West
	_aux = hw*hw*hw*hw;
	_aux = _sqrt(2)*hw / _sqrt(_aux + _max(_aux, eps));
	Numeric uw = _aux*huw, vw = _aux*hvw;

	//North
	_aux = hn*hn*hn*hn;
	_aux = _sqrt(2)*hn / _sqrt(_aux + _max(_aux, eps));
	Numeric un = _aux*hun, vn = _aux*hvn;

	//South
	_aux = hs*hs*hs*hs;
	_aux = _sqrt(2)*hs / _sqrt(_aux + _max(_aux, eps));
	Numeric us = _aux*hus, vs = _aux*hvs;

	//Write results to the global memory
	Numeric4 _Te;
	_Te.x = he; _Te.y = ue; _Te.z = ve; _Te.w = we;
	Numeric4 _Tw;
	_Tw.x = hw; _Tw.y = uw; _Tw.z = vw; _Tw.w = ww;
	Numeric4 _Tn;
	_Tn.x = hn; _Tn.y = un; _Tn.z = vn; _Tn.w = wn;
	Numeric4 _Ts;
	_Ts.x = hs; _Ts.y = us; _Ts.z = vs; _Ts.w = ws;
	__set_block_element__(Te, block_x, block_y, thread_x + 1, thread_y, grid_width + 1, _Te);
	__set_block_element__(Tw, block_x, block_y, thread_x, thread_y, grid_width + 1, _Tw);
	__set_block_element__(Tn, block_x, block_y, thread_x, thread_y + 1, grid_width, _Tn);
	__set_block_element__(Ts, block_x, block_y, thread_x, thread_y, grid_width, _Ts);
}


//CUDA kernel that computes boundary values for cardinal planes Tn and Ts
__global__ void __set_eow_woe_bc__(Numeric4* Te, Numeric4* Tw, const Numeric4* eow, const Numeric4* woe, const Numeric* B, Numeric eps,
	unsigned int grid_width, unsigned int grid_height)
{
	//Thread id
	int id = blockIdx.x * 4 * SVS_CUDA_BLOCK_SIZE + threadIdx.x;

	//Spill-over logics
	if (id >= grid_height) return;


	//east-of west
	Numeric4 _eow = eow[id];    //current east-of-west boundary value
	Numeric h = _eow.x - B[(1 + 2 * id)*(2 * grid_width + 1)];    //water height;
	Numeric _aux = h*h*h*h;
	_aux = _sqrt(2)*h / _sqrt(_aux + _max(_aux, eps));
	Numeric4 _Te;
	_Te.x = h;
	_Te.y = _aux*_eow.y;
	_Te.z = _aux*_eow.z;
	_Te.w = _eow.x;

	//Write result to memory
	Te[id*(grid_width + 1)] = _Te;


	//west-of-east
	Numeric4 _woe = woe[id];    //current west-of-east boundary value
	h = _woe.x - B[(1 + 2 * id)*(2 * grid_width + 1) + 2 * grid_width];    //water height
	_aux = h*h*h*h;
	_aux = _sqrt(2)*h / _sqrt(_aux + _max(_aux, eps));
	Numeric4 _Tw;
	_Tw.x = h;
	_Tw.y = _aux*_woe.y;
	_Tw.z = _aux*_woe.z;
	_Tw.w = _woe.x;

	//Write result to memory
	Tw[id*(grid_width + 1) + grid_width] = _Tw;
}


//CUDA kernel that computes boundary values for cardinal planes Tn and Ts
__global__ void __set_nos_son_bc__(Numeric4* Tn, Numeric4* Ts, const Numeric4* nos, const Numeric4* son, const Numeric* B, Numeric eps,
	unsigned int grid_width, unsigned int grid_height, bool reconstructed_nos, bool reconstructed_son)
{
	//Thread id
	int id = blockIdx.x * 4 * SVS_CUDA_BLOCK_SIZE + threadIdx.x;

	//Spill-over logics
	if (id >= grid_width) return;


	//north-of-south
	Numeric4 _nos = nos[id];
	if(!reconstructed_nos)
	{
		Numeric h = _nos.x - B[1 + 2 * id];    //water height
		Numeric _aux = h*h*h*h;
		_aux = _sqrt(2)*h / _sqrt(_aux + _max(_aux, eps));
		Numeric4 _Tn;
		_Tn.x = h;
		_Tn.y = _aux*_nos.y;
		_Tn.z = _aux*_nos.z;
		_Tn.w = _nos.x;

		//Write result to memory
		Tn[id] = _Tn;
	}
	else
	{
		Tn[id] = _nos;
	}
	


	//south-of-north
	Numeric4 _son = son[id];
	if(!reconstructed_son)
	{
		Numeric h = _son.x - B[(2 * grid_height)*(2 * grid_width + 1) + (1 + 2 * id)];
		Numeric _aux = h*h*h*h;
		_aux = _sqrt(2)*h / _sqrt(_aux + _max(_aux, eps));
		Numeric4 _Ts;
		_Ts.x = h;
		_Ts.y = _aux*_son.y;
		_Ts.z = _aux*_son.z;
		_Ts.w = _son.x;

		//Write result to memory
		Ts[grid_height*grid_width + id] = _Ts;
	}
	else
	{
		Ts[grid_height*grid_width + id] = _son;
	}
	
}


//CUDA kernel that computes the right-hand side of the related ODE system
__global__ void __compute_rhs__(Numeric4* rhs, const Numeric4* Te, const Numeric4* Tw, const Numeric4* Tn, const Numeric4* Ts,
	const Numeric4* U, const Numeric* B, Numeric dx, Numeric dy, Numeric g, unsigned int grid_width, unsigned int grid_height)
	//Block U already contains boundary conditions, but supplied pointer must point to the upper-left corner of the "internal" area
	//Variable T=(h,u,v,w) (cardinal direction literals are omitted)
	//Here Te, Tw, Tn and Ts contain corresponding boundary conditions.
	//Eastern boundary values must be contained in the first column of Te.
	//Western boundary conditions must be contained in the last column of Tw.
	//Northern boundary conditions must be contained in the first row of Tn.
	//Southern boundary conditions must be contained in the last row of Ts.

{
	//Get block address
	int block_x = blockIdx.x;
	int block_y = blockIdx.y;

	//Get thread address
	int thread_x = threadIdx.x;
	int thread_y = threadIdx.y;

	//Get absolute indexes of the thread
	int idx_x = block_x*SVS_CUDA_BLOCK_SIZE + thread_x;
	int idx_y = block_y*SVS_CUDA_BLOCK_SIZE + thread_y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;

	//Load data block to the shared memory
	__shared__ Numeric4 _Te[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric4 _Tw[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric4 _Tn[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];
	__shared__ Numeric4 _Ts[SVS_CUDA_BLOCK_SIZE][SVS_CUDA_BLOCK_SIZE];

	//Each thread loads single element from the global memory
	_Te[thread_y][thread_x] = __get_block_element__(Te, block_x, block_y, thread_x + 1, thread_y, grid_width + 1);
	_Tw[thread_y][thread_x] = __get_block_element__(Tw, block_x, block_y, thread_x, thread_y, grid_width + 1);
	_Tn[thread_y][thread_x] = __get_block_element__(Tn, block_x, block_y, thread_x, thread_y + 1, grid_width);
	_Ts[thread_y][thread_x] = __get_block_element__(Ts, block_x, block_y, thread_x, thread_y, grid_width);
	
	//Synchronize threads
	__syncthreads();


	//"Current" components (j+1/2,k) and (j,k+1/2)
	//Compute numerical fluxes
	Numeric3 c_Hx = __flux_Hx__(_Te[thread_y][thread_x], (thread_x == SVS_CUDA_BLOCK_SIZE - 1 || idx_x == grid_width - 1) ?
		__get_block_element__(Tw, block_x, block_y, thread_x + 1, thread_y, grid_width + 1) : _Tw[thread_y][thread_x + 1], g);
	
	Numeric3 c_Hy = __flux_Hy__(_Tn[thread_y][thread_x], (thread_y == SVS_CUDA_BLOCK_SIZE - 1 || idx_y == grid_height - 1) ?
		__get_block_element__(Ts, block_x, block_y, thread_x, thread_y + 1, grid_width) : _Ts[thread_y + 1][thread_x], g);

	//"Previous" components (j-1/2,k) and (j,k-1/2)
	//Compute numerical fluxes
	Numeric3 p_Hx = __flux_Hx__(thread_x == 0 ? __get_block_element__(Te, block_x, block_y, thread_x, thread_y, grid_width + 1) :
		_Te[thread_y][thread_x - 1], _Tw[thread_y][thread_x], g);
	
	Numeric3 p_Hy = __flux_Hy__(thread_y == 0 ? __get_block_element__(Tn, block_x, block_y, thread_x, thread_y, grid_width) :
		_Tn[thread_y - 1][thread_x], _Ts[thread_y][thread_x], g);

	//Compute the right-hand side of the ODE system induced by the "method of lines"
	Numeric4 _rhs;
	_rhs.x = -(c_Hx.x - p_Hx.x) / dx - (c_Hy.x - p_Hy.x) / dy;
	_rhs.y = -(c_Hx.y - p_Hx.y) / dx - (c_Hy.y - p_Hy.y) / dy;
	_rhs.z = -(c_Hx.z - p_Hx.z) / dx - (c_Hy.z - p_Hy.z) / dy;


	//Compute external force component
	//Account for the extra phenomenas here if needed
	Numeric3 S;
	S.x = 0;
	Numeric Bx = (_Te[thread_y][thread_x].w - _Tw[thread_y][thread_x].w + _Tw[thread_y][thread_x].x - _Te[thread_y][thread_x].x) / dx;
	Numeric By = (_Tn[thread_y][thread_x].w - _Ts[thread_y][thread_x].w + _Ts[thread_y][thread_x].x - _Tn[thread_y][thread_x].x) / dy;
	Numeric _aux = __get_block_element__(U, block_x, block_y, thread_x, thread_y, grid_width + 2).x -
		*__get_topography_block_element_addr__(B, block_x, block_y, thread_x, thread_y, grid_width);
	S.y = -g*_aux*Bx;
	S.z = -g*_aux*By;

	//Update rhs
	_rhs.y += S.y;
	_rhs.z += S.z;

	//Write results to the global memory
	__set_block_element__(rhs, block_x, block_y, thread_x, thread_y, grid_width, _rhs);
}


__global__ void __set_west_east_bc__(Numeric4* U, const Numeric4* wb, const Numeric4* eb, unsigned int grid_height, unsigned int grid_width)
{
	//Get thread id
	int id = blockIdx.x*(4 * SVS_CUDA_BLOCK_SIZE) + threadIdx.x;

	//Spill-over logics
	if (id >= grid_height) return;

	//Compute current row offset
	unsigned int row_offset = (id + 1)*(grid_width + 2);

	//Write boundaries to memory
	U[row_offset] = wb[id];
	U[row_offset + grid_width + 1] = eb[id];
}


__global__ void __set_south_bc__(Numeric4* U, const Numeric4* sb, unsigned int grid_width)
{
	//Get thread id
	int id = blockIdx.x*(4 * SVS_CUDA_BLOCK_SIZE) + threadIdx.x;

	//Spill-over logics
	if (id >= grid_width) return;

	//Write boundary to memory
	U[id + 1] = sb[id];
}


__global__ void __set_north_bc__(Numeric4 *U, const Numeric4 *nb, unsigned int grid_height, unsigned int grid_width)
{
	//Get thread id
	int id = blockIdx.x*(4 * SVS_CUDA_BLOCK_SIZE) + threadIdx.x;

	//Spill-over logics
	if (id >= grid_width) return;

	//Write boundary to memory
	U[(grid_height + 1)*(grid_width + 2) + id + 1] = nb[id];
}


__global__ void __euler__(Numeric4* U0, const Numeric4* rhs_U0, Numeric dt, unsigned int grid_height, unsigned int grid_width)
{
	//Retrieve absolute indexes of the thread
	int idx_x = blockIdx.x*SVS_CUDA_BLOCK_SIZE + threadIdx.x;
	int idx_y = blockIdx.y*SVS_CUDA_BLOCK_SIZE + threadIdx.y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;

	int offset = (idx_y + 1)*(grid_width + 2) + idx_x + 1;	//compute memory offset of the element from U0 corresponding to the current thread
	Numeric4 u0 = U0[offset];	//load the U0 element to register memory
	Numeric4 rhs = rhs_U0[idx_y*grid_width + idx_x];	//load the RHS(U0) element to register memory

	//Make an Euler step and write the result back to the global memory
	u0.x = u0.x + dt*rhs.x; u0.y = u0.y + dt*rhs.y; u0.z = u0.z + dt*rhs.z;
	U0[offset] = u0;
}


__global__ void __ssprk22_2__(Numeric4* U1, const Numeric4* U0, const Numeric4* rhs_U1, Numeric dt, unsigned int grid_height, unsigned int grid_width)
{
	//Retrieve absolute indexes of the thread
	int idx_x = blockIdx.x*SVS_CUDA_BLOCK_SIZE + threadIdx.x;
	int idx_y = blockIdx.y*SVS_CUDA_BLOCK_SIZE + threadIdx.y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;

	int offset = (idx_y + 1)*(grid_width + 2) + idx_x + 1;	//compute memory offset of the element from U0 corresponding to the current thread
	Numeric4 u0 = U0[offset];	//load the U0 element to register memory
	Numeric4 u1 = U1[offset];	//load the U1 element to register memory
	Numeric4 rhs = rhs_U1[idx_y*grid_width + idx_x];	//load the RHS(U1) element to register memory

	//Make the second semi-step of SSP-RK(2,2) and store the result in global memory
	u0.x = 0.5*u0.x + 0.5*u1.x + 0.5*dt*rhs.x;
	u0.y = 0.5*u0.y + 0.5*u1.y + 0.5*dt*rhs.y;
	u0.z = 0.5*u0.z + 0.5*u1.z + 0.5*dt*rhs.z;
	U1[offset] = u0;
}


__global__ void __ssprk33_2__(Numeric4* U1, const Numeric4* U0, const Numeric4* rhs_U1, Numeric dt, unsigned int grid_height, unsigned int grid_width)
{
	//Retrieve absolute indexes of the thread
	int idx_x = blockIdx.x*SVS_CUDA_BLOCK_SIZE + threadIdx.x;
	int idx_y = blockIdx.y*SVS_CUDA_BLOCK_SIZE + threadIdx.y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;

	int offset = (idx_y + 1)*(grid_width + 2) + idx_x + 1;	//compute memory offset of the element from U1 corresponding to the current thread
	Numeric4 u0 = U0[offset];	//load the U0 element to register memory
	Numeric4 u1 = U1[offset];	//load the U1 element to register memory
	Numeric4 rhs = rhs_U1[idx_y*grid_width + idx_x];	//load the RHS(U1) element to register memory

	//Make the second semi-step of the SSP-RK(3,3) and write the result to global memory
	u1.x = 0.75*u0.x + 0.25*u1.x + 0.25*dt*rhs.x;
	u1.y = 0.75*u0.y + 0.25*u1.y + 0.25*dt*rhs.y;
	u1.z = 0.75*u0.z + 0.25*u1.z + 0.25*dt*rhs.z;
	U1[offset] = u1;
}


__global__ void __ssprk33_3__(Numeric4* U2, const Numeric4* U0, const Numeric4* rhs_U2, Numeric dt, unsigned int grid_height, unsigned int grid_width)
{
	//Retrieve absolute indexes of the thread
	int idx_x = blockIdx.x*SVS_CUDA_BLOCK_SIZE + threadIdx.x;
	int idx_y = blockIdx.y*SVS_CUDA_BLOCK_SIZE + threadIdx.y;

	//Spill-over logics
	if (idx_x >= grid_width) return;
	if (idx_y >= grid_height) return;

	int offset = (idx_y + 1)*(grid_width + 2) + idx_x + 1;	//compute memory offset of the element from U2 corresponding to the current thread
	Numeric4 u2 = U2[offset];	//load element from U2 corresponding to the current thread
	Numeric4 u0 = U0[offset];	//load element from U0 corresponding to the current thread
	Numeric4 rhs = rhs_U2[idx_y*grid_width + idx_x];	//load element from RHS(U2) corresponding to the current thread

	//Make the last semi-step of SP-RK(3,3) and write the result to global memory
	u2.x = (1.0 / 3.0)*u0.x + (2.0 / 3.0)*u2.x + (2.0 / 3.0)*dt*rhs.x;
	u2.y = (1.0 / 3.0)*u0.y + (2.0 / 3.0)*u2.y + (2.0 / 3.0)*dt*rhs.y;
	u2.z = (1.0 / 3.0)*u0.z + (2.0 / 3.0)*u2.z + (2.0 / 3.0)*dt*rhs.z;
	U2[offset] = u2;
}
//***********************************************************************************************************************************************
