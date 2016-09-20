#ifndef TW__GENERAL_TEXTURE_TYPES__

namespace tiny_world
{
	//Implements generalized texture pointer type, which can encapsulate both immutable and buffer texture pointers
	class TexturePointer
	{
	private:
		void* p_texture_pointer;	//stores abstract texture pointer

	protected:
		TexturePointer(void* p_pointer = nullptr);	//initializes the pointer object using the value provided

		void setPointerValue(void* p_pointer);	//sets new value for the encapsulated pointer
		void* getPointerValue() const;	//returns the actual value of the encapsulated pointer

	public:
		operator bool() const;	//returns 'true' if encapsulated pointer is not equal to 'nullptr'. Returns 'false' otherwise
	};
}

#define TW__GENERAL_TEXTURE_TYPES__
#endif