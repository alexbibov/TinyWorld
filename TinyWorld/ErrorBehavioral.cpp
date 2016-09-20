#include "ErrorBehavioral.h"

using namespace tiny_world;


void ErrorBehavioral::call_error_callback(const std::string& msg) const{ error_callback(msg); }

void ErrorBehavioral::call_error_callback() const { error_callback(error_string); }

bool ErrorBehavioral::set_error_state(bool state) const
{
	bool rv = error_state;
	error_state = state;
	return rv;
}

void ErrorBehavioral::set_error_string(const std::string& err_msg) const{ error_string = err_msg; }



ErrorBehavioral::ErrorBehavioral() : error_state{ false }, error_string(""), error_callback{ [](const std::string& err_msg)->void{} }
{

}

ErrorBehavioral::ErrorBehavioral(ErrorBehavioral&& other) : error_state{ other.error_state }, 
error_string(std::move(other.error_string)), error_callback{ std::move(other.error_callback) }
{

}

ErrorBehavioral& ErrorBehavioral::operator=(ErrorBehavioral&& other)
{
	if (this == &other)
		return *this;

	error_state = other.error_state;
	error_string = std::move(error_string);
	error_callback = std::move(other.error_callback);
	return *this;
}

ErrorBehavioral::~ErrorBehavioral()
{

}


void ErrorBehavioral::registerErrorCallback(std::function<void(const std::string& err_msg)> error_callback)
{
	this->error_callback = error_callback;
}

bool ErrorBehavioral::resetErrorState()
{
	bool rv = error_state;
	error_state = false;
	error_string = "";
	return rv;
}

bool ErrorBehavioral::getErrorState() const { return error_state; }

const char* ErrorBehavioral::getErrorString() const { return error_string.c_str(); }

ErrorBehavioral::operator bool() const { return !error_state; }

bool ErrorBehavioral::operator!() const { return error_state; }