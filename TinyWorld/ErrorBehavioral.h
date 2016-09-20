//Implements standard error infrastructure for the classes that can exist in erroneous state

#ifndef TW__ERROR_BEHAVIORAL__

#include <string>
#include <functional>


namespace tiny_world
{
	class ErrorBehavioral
	{
	private:
		//Error status of the object. Equals 'true' if the object is in an erroneous state. An object can be put into
		//an erroneous state if one of the preceding operations involving this object has failed.
		mutable bool error_state;

		//String describing the last error occurred in the object. The string provides description of the error
		//in human-friendly manner. However, it provides no data about errors that might have occurred before the last one.
		mutable std::string error_string;

		//Callback function that is called when the object enters an erroneous state.
		std::function<void(const std::string& err_msg)> error_callback;

	protected:
		void call_error_callback(const std::string& msg) const;	//calls error callback with error message referred by msg
		void call_error_callback() const;	//calls error callback using the error string as the source for the error message
		bool set_error_state(bool state) const;	//sets error state of the object, returns previous error state
		void set_error_string(const std::string& err_msg) const;	//sets error string describing the last occurred error

		//Default constructor. Initializes object with error state set to 'false' and with empty error string
		ErrorBehavioral();

		//Copy constructor
		ErrorBehavioral(const ErrorBehavioral& other) = default;

		//Move constructor (move constructor is not defaulted for compatibility reasons as Microsoft C++ compiler v12 does not support it)
		ErrorBehavioral(ErrorBehavioral&& other);

		//Copy assignment operator
		ErrorBehavioral& operator=(const ErrorBehavioral& other) = default;

		//Move assignment operator (move-assignment is not defaulted for compatibility reasons as Microsoft C++ compiler v12 does not support it)
		ErrorBehavioral& operator=(ErrorBehavioral&& other);

	public:
		//Registers new error callback. Error callback is a function, which is called immediately when the object enters 
		//erroneous state. This facility is useful when it is needed to continuously log error status of the object.
		void registerErrorCallback(std::function<void(const std::string& err_msg)> error_callback);

		//Resets error state of the object to 'false' and returns the previous value of the error state
		bool resetErrorState();

		//Returns current error state of the object
		bool getErrorState() const;

		//Returns textual description of the last error occurred
		const char* getErrorString() const;

		//Returns 'true' if object is NOT in an erroneous state. Returns 'false' otherwise. 
		operator bool() const;

		//Returns 'true' if object IS in erroneous state. Returns 'false' otherwise.
		//This function is equivalent to getErrorState()
		bool operator!() const;


		//Destructor
		virtual ~ErrorBehavioral();
	};
}


#define TW__ERROR_BEHAVIORAL__
#endif