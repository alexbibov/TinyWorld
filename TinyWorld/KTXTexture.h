#ifndef TW__KTX_TEXTURE__

//Class KTXReader provides a simple but all-featured loader of Khronos texture format that is recommended for use with OpenGL

#include <fstream>
#include <string>
#include <array>
#include <map>
#include <stdint.h>

#include <GL/glew.h>

#include "ImmutableTexture.h"
#include "ErrorBehavioral.h"

//Converts generic interface texture representation to a 1D texture
#define TEXTURE_1D(texture)\
	(*dynamic_cast<tiny_world::ImmutableTexture1D*>(texture))

//Converts generic interface texture representation to a 2D texture
#define TEXTURE_2D(texture)\
	(*dynamic_cast<tiny_world::ImmutableTexture2D*>(texture))

//Converts generic interface texture representation to a 3D texture
#define TEXTURE_3D(texture)\
	(*dynamic_cast<tiny_world::ImmutableTexture3D*)(texture))

//Converts generic interface texture representation to a cubemap texture
#define TEXTURE_CUBEMAP(texture)\
	(*dynamic_cast<tiny_world::ImmutableTextureCubeMap*>(texture))

namespace tiny_world{

	class KTXTexture final : public ErrorBehavioral{
	private:
		enum class Endianness{ BIG_ENDIAN, LITTLE_ENDIAN };
		enum class MachineWordSize : uint32_t{ _16_BIT = 2, _32_BIT = 4 };

		union single_word32{
			uint32_t integer;
			char byte_storage[4];
		};

		union single_word16{
			uint16_t integer;
			char byte_storage[2];
		};

		class KeyValueElem{
		private:
			size_t elem_size;
			void *elem_data;
		public:
			KeyValueElem(size_t elem_size, const void* elem_data);
			KeyValueElem(const KeyValueElem& other);
			KeyValueElem(KeyValueElem&& other);
			~KeyValueElem();

			bool operator == (const KeyValueElem& other) const;
			bool operator < (const KeyValueElem& other) const;
			bool operator <= (const KeyValueElem& other) const;
			bool operator > (const KeyValueElem& other) const;
			bool operator >= (const KeyValueElem& other) const;
			bool operator != (const KeyValueElem& other) const;
			KeyValueElem& operator = (const KeyValueElem& other);
			KeyValueElem& operator = (KeyValueElem&& other);

			size_t size() const;
			const void* data() const;
		};

		bool initialization_status;	//equals "true" if KTX data has been loaded from a file and function releaseTexture() has not been called afterwards
		bool endianness_conversion_required;	//set if endianness of the executing environment differs from endianness of KTX's creator
		std::string file_name;	//location of the source KTX-file within the host file system
		std::filebuf ktx_input_buffer;	//stream buffer used to read the KTX source
		ImmutableTexture *p_contained_texture;	//pointer to contained immutable texture

		//********************************************************KTX data particulars************************************************************

		//Endianness of machine created the KTX data
		Endianness ktx_creator_endianness;

		//Ordered map that holds scalar preamble of the KTX data structure
		std::map<std::string, uint32_t> ktx_scalar_preamble;

		//Ordered map that holds user-defined key-value pairs
		std::map<KeyValueElem, KeyValueElem> ktx_key_value_data;
		//****************************************************************************************************************************************

		//Helper internal functions

		bool extractPreambleScalarUint32(std::string parameter_name);	//extracts scalar uint32_t parameter from KTX-file preamble
		inline Endianness getPlatformEndianness() const;	//returns endianness of the executing platform
		inline uint32_t swapWord32(uint32_t word32) const;	//Swaps byte order in a 32-bit word
		inline uint16_t swapWord16(uint16_t word16) const;	//Swaps byte order in a 16-bit word
		void* convertBufferEndianness(void* buffer, size_t buffer_size, MachineWordSize word_size) const;		//Converts data in buffer referred by pointer "buffer" using machine word size "word_size". Conversion results are written into the SAME buffer.

	public:
		KTXTexture();	//Default constructor
		~KTXTexture();	//Class destructor

		bool loadTexture(std::string file_name);	//Reads KTX data from file located at directory defined by file_name
		ImmutableTexture* getContainedTexture() const;	//Returns pointer to the contained texture or nullptr if no texture was loaded
		void releaseTexture();	//Releases currently loaded texture. If no texture was loaded, this operation has no effect.
	};

}

#define TW__KTX_TEXTURE__
#endif