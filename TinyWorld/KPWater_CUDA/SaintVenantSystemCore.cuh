#ifndef SVS__CORE__
#define SVS__CORE__

#include <vector>
#include <map>
#include <functional>
#include <omp.h>

#include "SaintVenantSystemCommon.cuh"
#include "CudaErrorWrapper.h"
#include "CudaDeviceScheduler.h"
#include "UnifiedMemoryBlock.h"


namespace SaintVenantSystem
{
	class SVSCore final
	{
	public:
		//Describes settings of the computation domain
		struct DomainSettings
		{
			unsigned int width, height;
			Numeric dx, dy;
		};


		//Default constructor: create object without initialization
		SVSCore();

		//Copy constructor
		explicit SVSCore(const SVSCore &other);

		//Move constructor
		//explicit SVSCore(SVSCore&& other);

		//Copy assignment operator
		SVSCore& operator = (const SVSCore& other);

		//Move assignment operator
		//SVSCore& operator=(SVSCore&& other);

		//Initializing constructor
		SVSCore(const SVSVarU init_state, unsigned int height, unsigned int width, Numeric dx, Numeric dy, Numeric g, Numeric theta, Numeric eps, const Numeric *topography);

		//Auxiliary initializing constructor
		SVSCore(const SVSVarU init_state, SVSParameters opts);

		//Default destructor
		~SVSCore();

		//Returns 'true' if object is NOT in an erroneous state. Returns 'false' otherwise
		operator bool() const;

		//Initialize Saint-Venant system
		void initialize(const SVSVarU init_state, unsigned int height, unsigned int width, Numeric dx, Numeric dy, Numeric g, Numeric theta, Numeric eps, const Numeric *topography);
		void initialize(const SVSVarU init_state, SVSParameters opts);

		//Resets SVSCore object to its initial uninitialized state. Releases all the memory associated with the object.
		void reset();

		//Gets average time spent by each device installed on the host systems to execute CUDA kernels during the last invocation of ComputeRHS(...)
		float getCudaExecutionTime() const;

		//Returns a pair with the first element containing descriptor of the last error occurred and the second element equal to the numerical identifier of the
		//CUDA-capable device that caused the error.
		std::pair<CudaUtils::CudaErrorWrapper, int> getLastError() const;

		//Returns 'true' if object IS in an erroneous state, returns 'false' otherwise
		bool getErrorState() const;

		//Checks if the object has been initialized
		bool isInitialized() const;

		//Registers error callback function, which is called when the object enters erroneous state
		void registerErrorCallback(const std::function<void(const CudaUtils::CudaErrorWrapper&, int)>& error_callback);

		//Implements SSP-RK(3,3) based solver
		void solveSSPRK33(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt);
		//Implements SSP-RK(2,2) based solver
		void solveSSPRK22(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt);
		//Implements Euler method based solver
		void solveEuler(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb, Numeric dt);

		//Returns pointer to CPU memory containing current system state. A call to this function causes data transfer from GPU to CPU memory.
		//Note that array returned by this function is in raw format, i.e. it contains the boundary values, which means that assuming an N-by-M compute domain the function
		//will return (N+2)*(M+2) values arranged in row-major order.
		const Numeric4 *getSystemState() const;

		//Sets gravity constant to be used by the solver
		void setGravityConstant(Numeric g);

		//Returns value of the gravity constant currently used by the solver
		Numeric getGravityConstant() const;

		//Sets "theta" parameter of the solver. The value must be in range [1,2], where the greater values correspond to less dissipative solutions
		void setTheta(Numeric theta);

		//Returns value of the "theta" parameter of the solver
		Numeric getTheta() const;

		//Sets "epsilon" parameter of the solver. The value of this parameter affects quality of the desingularizers. The recommended value is max{dx^4, dy^4}
		void setEpsilon(Numeric epsilon);

		//Returns value of the "epsilon" parameter of the solver
		Numeric getEpsilon() const;

		//Returns settings of the discretization domain
		DomainSettings getDomainSettings() const;

		//Updates current system state using user-supplied value
		void updateSystemState(SVSVarU new_system_state);

		//Updates topography using user-supplied value
		void updateTopographyData(const Numeric* B);

	private:
		typedef std::map<uint32_t, CudaUtils::GPUMemoryBlock> GPUMemoryBlock_Map;
		typedef std::vector<cudaEvent_t> cudaEvent_t_Vector;
		typedef std::vector<Numeric*> Numericpointer_Vector;
		typedef std::vector<Numeric4*> Numeric4pointer_Vector;
		typedef std::vector<unsigned int> unsigned_int_Vector;
		typedef std::vector<cudaStream_t> cudaStream_t_Vector;

		//Pointer to the CPU memory block containing current state of the SV-system
		Numeric4 *cpu_system_state;

		//Boundary values at the bounding cell centers
		Numeric4 *cpu_wb, *cpu_eb, *cpu_sb, *cpu_nb;

		//Boundary values at the bounding cell edges
		Numeric4 *cpu_eow, *cpu_woe, *cpu_nos, *cpu_son;

		//Temporary buffer used to exchange boundary condition values between the devices
		Numeric4 *cpu_bc_exchange_buf;

		unsigned_int_Vector st_grid_height;	//Number of grid rows associated with each CUDA-device
		cudaStream_t_Vector st_dev_stream;  //Main CUDA command stream related to each device in use 

		CudaUtils::CPUMemoryBlock memCPU;	//CPU linear memory buffer for I/O
		GPUMemoryBlock_Map memGPU;	//GPU linear memory buffer for I/O (such buffer is created for each GPU in the system that supports CUDA)

		//CUDA event recorded when the first and the second kernels are completed. 
		//Needed for synchronization purposes.
		cudaEvent_t_Vector st_ctxc;

		//CUDA event recorded at the beginning of each subtask.
		cudaEvent_t_Vector st_start;

		//CUDA event recorded at the end of each subtask.
		cudaEvent_t_Vector st_finish;


		mutable CudaUtils::CudaDeviceScheduler CUDA_device_scheduler;	//task scheduler needed to distribute the workload between CUDA-capable devices
		int CUDA_device_count;	//number of CUDA-capable devices installed on the host system
		const uint32_t best_device_id;	//identifier of the "best" CUDA-capable device available on the system
		const uint32_t worst_device_id;	//identifier of the "worst" CUDA-capable device available on the system
		mutable CudaUtils::CudaErrorWrapper error_descriptor;    //descriptor of the last error occurred
		mutable CudaUtils::CudaErrorWrapper aux_err_desc;	//auxiliary error descriptor used to receive error information from a particular function call
		mutable bool error_state;	//equals 'true' if object resides in an erroneous state, equals 'false' otherwise 

		//Callback function, which is called when the object gets into an erroneous state. The function accepts two arguments, which are 
		//in order, the error wrapping object and numerical identifier of the device, which has emitted the error.
		std::function<void(const CudaUtils::CudaErrorWrapper&, int)> error_callback;

		//Contains numerical identifier of the device, which caused the last error. If the last error is not related to a concrete device, equals to -1. 
		mutable int error_source;

		bool is_initialized;    //states if the object has been initialized
		SVSParameters opts;  //options specifying behavior of the SV-system, its discretization grid, and the other parameters of the KP-scheme
		float exec_time;	//the GPU time consumed during the last evaluation of ComputeRHS(...)


		//Device memory pointers

		Numeric4pointer_Vector gpu_Te, gpu_Tw, gpu_Ts, gpu_Tn,	//cardinal values
			gpu_eow, gpu_woe, gpu_nos, gpu_son,	//cell edge boundary values
			gpu_wb, gpu_eb;	//west and east wall boundary values
		Numeric4pointer_Vector gpu_rhs;	//memory block needed to store result of evaluation of ComputeRHS(...)
		Numeric4pointer_Vector gpu_system_state;	//vector of GPU memory pointers referring to the current system state data distributed between CUDA-devices
		Numeric4pointer_Vector gpu_temp;	//temporary buffer used by SSP-RK(2,2) and SSP-RK(2,3)
		Numeric4 *gpu_sb, *gpu_nb;	//south and north wall boundary values
		Numericpointer_Vector gpu_B;	//bottom topography


		//IMPORTANT! The next three functions are just helpers meant to simplify implementation of the solvers. 
		//The helper functions distribute the corresponding tasks between devices THEMSELVES, so none of them should be called
		//inside a task distribution loop.

		//Loads supplied boundary values into internal buffers
		void loadBoundaryValues(const SVSBoundary wb, const SVSBoundary eb, const SVSBoundary sb, const SVSBoundary nb);

		//Performs a deep copy of data from gpu_system_state to gpu_temp
		void cloneSystemState();

		//Implements spatial discretization of Saint-Venant system as defined by the KP-scheme. 
		//The spatial discretization operator is applied to the data stored in gpu_system state at the time of the call.
		//The result is stored in gpu_rhs buffer.
		void computeRHS();

		//This function is meant to perform cross-device reconstruction of boundary values when the system state gets evolved by a solver.
		//Implementation of this function is a workaround and is based on a hack: it is not allowed to use cudaMemcpy(Async) when one the
		//GPU memory addresses is not a multiple of 16. However, since state element is represented by Numeric4 the latter requirement is
		//satisfied for this particular system. Nevertheless, such hacks make the code oblique.
		void reconstructBoundaries();
	};

}


#endif
