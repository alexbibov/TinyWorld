#include <cstring>
#include <utility>
#include <ctgmath>

#include "KTXTexture.h"

#include "ImmutableTextureCubeMap.h"
#include "ImmutableTexture1D.h"
#include "ImmutableTexture2D.h"
#include "ImmutableTexture3D.h"

using namespace tiny_world;

KTXTexture::KTXTexture() : initialization_status(false),
endianness_conversion_required(false), file_name(""), p_contained_texture(nullptr)
{
}

KTXTexture::~KTXTexture()
{
	if (p_contained_texture)
		delete p_contained_texture;

	if (ktx_input_buffer.is_open())		//close input stream if it was open
		ktx_input_buffer.close();
}

bool KTXTexture::loadTexture(std::string file_name)
{
	if (initialization_status)
		return false;	//texture was already loaded

	this->file_name = file_name;	//update the file_name field of the object

	//Attach object's file buffer to the file system
	if (!ktx_input_buffer.open(file_name, std::ios_base::in | std::ios_base::binary))
	{
		set_error_state(true);
		std::string err_msg = "Can not open file " + file_name;
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//**********************Firstly, check presence and correctness of the KTX identifier*********************************

	typedef std::array<char, 12> KTXid;

	const KTXid ktx_identifier = { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
	KTXid input_file_identifier;

	if (ktx_input_buffer.sgetn(input_file_identifier.data(), 12) < 12 || input_file_identifier != ktx_identifier)
	{
		set_error_state(true);
		std::string err_msg = "File " + file_name + " is not a valid KTX file";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	//*************************************Check endianness of the file's creator*****************************************

	typedef std::array<char, 4>  endianness_id;

	const endianness_id big_endian = { 0x04, 0x03, 0x02, 0x01 };
	const endianness_id little_endian = { 0x01, 0x02, 0x03, 0x04 };
	endianness_id input_file_endianness;

	if (ktx_input_buffer.sgetn(input_file_endianness.data(), 4) < 4)
	{
		set_error_state(true);
		std::string err_msg = "Unable to read endianness of the file " + file_name + ". File is damaged";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	bool input_file_endianness_determined = false;

	if (input_file_endianness == big_endian){
		ktx_creator_endianness = Endianness::BIG_ENDIAN;
		input_file_endianness_determined = true;
	}

	if (input_file_endianness == little_endian){
		ktx_creator_endianness = Endianness::LITTLE_ENDIAN;
		input_file_endianness_determined = true;
	}

	if (!input_file_endianness_determined)
	{
		set_error_state(true);
		std::string err_msg = "Unable to determine endianness of the file " + file_name + " . The file is damaged or has incorrect format";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	endianness_conversion_required = getPlatformEndianness() != ktx_creator_endianness;

	//******************************************Extract the rest of the KTX scalar preamble*******************************
	if (!extractPreambleScalarUint32("glType")) return false;
	if (!extractPreambleScalarUint32("glTypeSize")) return false;
	if (!extractPreambleScalarUint32("glFormat")) return false;
	if (!extractPreambleScalarUint32("glInternalFormat")) return false;
	if (!extractPreambleScalarUint32("glBaseInternalFormat")) return false;
	if (!extractPreambleScalarUint32("pixelWidth")) return false;
	if (!extractPreambleScalarUint32("pixelHeight")) return false;
	if (!extractPreambleScalarUint32("pixelDepth")) return false;
	if (!extractPreambleScalarUint32("numberOfArrayElements")) return false;
	if (!extractPreambleScalarUint32("numberOfFaces")) return false;
	if (!extractPreambleScalarUint32("numberOfMipmapLevels")) return false;
	if (!extractPreambleScalarUint32("bytesOfKeyValueData")) return false;

	//***************************************Extract user-defined key-value pair sequence*********************************
	char *keyvalue_data_buf = new char[ktx_scalar_preamble["bytesOfKeyValueData"]];
	if (ktx_input_buffer.sgetn(keyvalue_data_buf, ktx_scalar_preamble["bytesOfKeyValueData"]) < ktx_scalar_preamble["bytesOfKeyValueData"])
	{
		set_error_state(true);
		std::string err_msg = "Failure while reading user-defined key/value sequence from the KTX-file (" + file_name + ". File is damaged";
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	size_t offset = 0;
	while (offset < ktx_scalar_preamble["bytesOfKeyValueData"])
	{
		//read full size of a key-value pair
		single_word32 keyAndValueByteSize;
		keyAndValueByteSize.byte_storage[0] = (keyvalue_data_buf + offset)[0];
		keyAndValueByteSize.byte_storage[1] = (keyvalue_data_buf + offset)[1];
		keyAndValueByteSize.byte_storage[2] = (keyvalue_data_buf + offset)[2];
		keyAndValueByteSize.byte_storage[3] = (keyvalue_data_buf + offset)[3];
		keyAndValueByteSize.integer =
			endianness_conversion_required ? swapWord32(keyAndValueByteSize.integer) : keyAndValueByteSize.integer;

		//read key and value
		size_t key_actual_length = std::char_traits<char>::length(keyvalue_data_buf + offset + 4);
		KeyValueElem key(key_actual_length + 1, keyvalue_data_buf + offset + 4);
		KeyValueElem value(keyAndValueByteSize.integer - key.size(), keyvalue_data_buf + offset + 4 + key.size());
		ktx_key_value_data.insert(std::make_pair(key, value));

		//shift the offset
		offset += 4 + keyAndValueByteSize.integer + (3 - (keyAndValueByteSize.integer + 3) % 4);
	}
	delete[] keyvalue_data_buf;


	//***********************************************Read texture data****************************************************

	//Initialize texture size and number of mipmap levels
	TextureSize tex_size;
	tex_size.width = ktx_scalar_preamble["pixelWidth"];
	tex_size.height = ktx_scalar_preamble["pixelHeight"];
	tex_size.depth = ktx_scalar_preamble["pixelDepth"];
	uint32_t number_of_mipmap_levels = ktx_scalar_preamble["numberOfMipmapLevels"] ? ktx_scalar_preamble["numberOfMipmapLevels"] :
		static_cast<uint32_t>(std::log2(std::max(ktx_scalar_preamble["pixelWidth"], ktx_scalar_preamble["pixelHeight"])));

	enum class texture_type{
		_1d, _2d, _3d, cubemap
	}tex_type;	//this will store the target type of the texture being loaded

	bool is_compressed = ktx_scalar_preamble["glFormat"] == 0;
	PixelFormatTraits pixel_format_traits{ ktx_scalar_preamble["glInternalFormat"] };

	//Non-array cube maps should be treated in a special way... 
	bool is_non_array_cube_map = ktx_scalar_preamble["numberOfFaces"] == 6 && ktx_scalar_preamble["numberOfArrayElements"] == 0;

	if (ktx_scalar_preamble["numberOfFaces"] == 6)
	{
		tex_type = texture_type::cubemap;	//texture type is a cubemap

		//Initialize cube map texture
		ImmutableTextureCubeMap* cubemap_tex = new ImmutableTextureCubeMap("KTX_cube_map_texture_" + file_name);


		//Allocate storage for the texture
		if (is_compressed)
			cubemap_tex->allocateStorage(number_of_mipmap_levels, ktx_scalar_preamble["numberOfArrayElements"], tex_size, 
			static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]));
		else
			cubemap_tex->allocateStorage(number_of_mipmap_levels, ktx_scalar_preamble["numberOfArrayElements"], tex_size, 
			static_cast<InternalPixelFormat>(ktx_scalar_preamble["glInternalFormat"]));

		offset = 0;
		for (unsigned int mipmap_level = 0; mipmap_level < (ktx_scalar_preamble["numberOfMipmapLevels"] ? ktx_scalar_preamble["numberOfMipmapLevels"] : 1); ++mipmap_level)
		{
			single_word32 imageSize;	//in this case imageSize is the size of a single cube face not including the padding values
			if (ktx_input_buffer.sgetn(imageSize.byte_storage, 4) < 4)
			{
				set_error_state(true);
				std::string err_msg = "Unable to read field \"imageSize\" in LOD-level section " +
					std::to_string(mipmap_level) + " from file " + file_name + ". File is damaged";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
			imageSize.integer = endianness_conversion_required ? swapWord32(imageSize.integer) : imageSize.integer;
			offset += 4;

			//Read data from the cubemap faces. The data is read in a slightly different manner for array and non-array cubemap textures
			if (is_non_array_cube_map)
			{
				char *face_data[6];
				for (int face = 0; face < 6; ++face)
				{
					face_data[face] = new char[imageSize.integer];
					if (ktx_input_buffer.sgetn(face_data[face], imageSize.integer) < imageSize.integer)
					{
						set_error_state(true);
						std::string err_msg = "Unable to read image data of a cube map texture in face " +
							std::to_string(face) + " from file" + file_name + ". File is damaged";
						set_error_string(err_msg);
						call_error_callback(err_msg);
						return false;
					}
					offset += imageSize.integer;

					char face_padding[3];
					int padding_count = 3 - (3 + imageSize.integer) % 4;
					if (ktx_input_buffer.sgetn(face_padding, padding_count) < padding_count)
					{
						set_error_state(true);
						std::string err_msg = "Invalid face padding value in cube map face " + std::to_string(face) +
							" while reading file " + file_name + ". File is damaged or is in invalid format";
						set_error_string(err_msg);
						call_error_callback(err_msg);
						return false;
					}
					offset += padding_count;


					//Perform endianness conversion if needed
					if (endianness_conversion_required)
					{
						switch (ktx_scalar_preamble["glTypeSize"])
						{
						case 1:
							//No conversion is needed
							break;
						case 2:
						case 4:
							convertBufferEndianness(face_data[face], imageSize.integer, static_cast<MachineWordSize>(ktx_scalar_preamble["glTypeSize"]));
							break;
						default:
							set_error_state(true);
							const char* err_msg = "Can not perform endianness conversion: size of the host machine's word is not recognized";
							set_error_string(err_msg);
							call_error_callback(err_msg);
							return false;
						}
					}
				}

				//Populate cubemap texture with data
				if (is_compressed)
					cubemap_tex->setMipmapLevelData(mipmap_level, static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]),
					imageSize.integer, face_data[0], face_data[1], face_data[2], face_data[3], face_data[4], face_data[5]);
				else
					cubemap_tex->setMipmapLevelData(mipmap_level, static_cast<PixelLayout>(ktx_scalar_preamble["glFormat"]),
					static_cast<PixelDataType>(ktx_scalar_preamble["glType"]), face_data[0], face_data[1], face_data[2], face_data[3], face_data[4], face_data[5]);

				for (int face = 0; face < 6; ++face) delete[] face_data[face];
			}
			else
			{
				//For cubemap arrays the data for all faces is read at once
				char *face_data = new char[imageSize.integer];
				if (ktx_input_buffer.sgetn(face_data, imageSize.integer) < imageSize.integer)
				{
					set_error_state(true);
					std::string err_msg = "Unable to read image data of an array cubemap texture from file " + file_name + ". File is damaged";
					set_error_string(err_msg);
					call_error_callback(err_msg);
					return false;
				}

				if (endianness_conversion_required)
				{
					switch (ktx_scalar_preamble["glTypeSize"])
					{
					case 1:
						//No conversion required
						break;
					case 2:
					case 4:
						convertBufferEndianness(face_data, imageSize.integer, static_cast<MachineWordSize>(ktx_scalar_preamble["glTypeSize"]));
						break;
					default:
						set_error_state(true);
						const char* err_msg = "Can not perform endianness conversion: size of the host machine's word is not recognized";
						set_error_string(err_msg);
						call_error_callback(err_msg);
						return false;
					}
				}


				//Populate cubemap texture with data
				if (is_compressed)
					cubemap_tex->setMipmapLevelMultiLayerFacesData(mipmap_level, 0, 6 * ktx_scalar_preamble["numberOfArrayElements"],
					static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]), imageSize.integer, face_data);
				else
					cubemap_tex->setMipmapLevelMultiLayerFacesData(mipmap_level, 0, 6 * ktx_scalar_preamble["numberOfArrayElements"],
					static_cast<PixelLayout>(ktx_scalar_preamble["glFormat"]), static_cast<PixelDataType>(ktx_scalar_preamble["glType"]), face_data);


				delete[] face_data;
			}


			//Account for mipmap padding
			char mipmap_padding[3];
			int padding_count = 3 - (6 * imageSize.integer + 3) % 4;
			if (ktx_input_buffer.sgetn(mipmap_padding, padding_count) < padding_count)
			{
				set_error_state(true);
				std::string err_msg = std::string("Invalid mipmap padding value found while reading from file ") +
					file_name + std::string(". File is damaged or is in invalid format");
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
			offset += padding_count;
		}
		p_contained_texture = cubemap_tex;
	}
	else
	{
		//If texture stored in KTX-file is not a non-array cube map it is read in the usual way...
		//But first we need to determine what kind of texture is stored in the KTX-file

		if (ktx_scalar_preamble["pixelHeight"] == 0 && ktx_scalar_preamble["pixelDepth"] == 0)
			tex_type = texture_type::_1d;

		if (ktx_scalar_preamble["pixelHeight"] != 0 && ktx_scalar_preamble["pixelDepth"] == 0)
			tex_type = texture_type::_2d;

		if (ktx_scalar_preamble["pixelHeight"] != 0 && ktx_scalar_preamble["pixelDepth"] != 0)
			tex_type = texture_type::_3d;

		//Allocate texture storage depending on the determined type
		switch (tex_type)
		{
		case texture_type::_1d:
			p_contained_texture = new ImmutableTexture1D(std::string("KTX_1D_texture_") + file_name);
			break;

		case texture_type::_2d:
			p_contained_texture = new ImmutableTexture2D(std::string("KTX_2D_texture_") + file_name);
			break;

		case texture_type::_3d:
			p_contained_texture = new ImmutableTexture3D(std::string("KTX_3D_texture_") + file_name);
			break;

		default:
			set_error_state(true);
			const char* err_msg = "KTX load error: unsupported texture type";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return false;
		}
		if (is_compressed)
			p_contained_texture->allocateStorage(number_of_mipmap_levels, ktx_scalar_preamble["numberOfArrayElements"], tex_size,
			static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]));
		else
			p_contained_texture->allocateStorage(number_of_mipmap_levels, ktx_scalar_preamble["numberOfArrayElements"], tex_size,
			static_cast<InternalPixelFormat>(ktx_scalar_preamble["glInternalFormat"]));



		size_t offset = 0;
		for (unsigned int mipmap_level = 0; mipmap_level < (ktx_scalar_preamble["numberOfMipmapLevels"] ? ktx_scalar_preamble["numberOfMipmapLevels"] : 1); ++mipmap_level)
		{
			single_word32 imageSize;	//here imageSize is the size of whole texture including all array layers
			if (ktx_input_buffer.sgetn(imageSize.byte_storage, 4) < 4)
			{
				set_error_state(true);
				std::string err_msg = std::string("Unable to read field \"imageSize\" in LOD-level ") +
					std::to_string(mipmap_level) + std::string(" from file ") + file_name +
					std::string(". File is damaged");
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
			offset += 4;

			char *img_data = new char[imageSize.integer];
			if (ktx_input_buffer.sgetn(img_data, imageSize.integer) < imageSize.integer)
			{
				set_error_state(true);
				std::string err_msg = std::string("Unable to extract image data while reading from file ") +
					file_name + std::string(". File is damaged");
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
			offset += imageSize.integer;

			//Perform endianness conversion if necessary
			if (endianness_conversion_required)
			{
				switch (ktx_scalar_preamble["glTypeSize"])
				{
				case 1:
					//No conversion is needed
					break;
				case 2:
				case 4:
					convertBufferEndianness(img_data, imageSize.integer, static_cast<MachineWordSize>(ktx_scalar_preamble["glTypeSize"]));
					break;
				default:
					set_error_state(true);
					const char* err_msg = "Can not perform endianness conversion: unknown platform machine word size";
					set_error_string(err_msg);
					call_error_callback(err_msg);
					return false;
				}
			}

			//Populate texture with data
			switch (tex_type)
			{
			case texture_type::_1d:
			{
				ImmutableTexture1D* p_1d_tex = dynamic_cast<ImmutableTexture1D*>(p_contained_texture);
				if (is_compressed)
					p_1d_tex->setMipmapLevelData(mipmap_level, static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]), imageSize.integer, img_data);
				else
					p_1d_tex->setMipmapLevelData(mipmap_level, static_cast<PixelLayout>(ktx_scalar_preamble["glFormat"]),
					static_cast<PixelDataType>(ktx_scalar_preamble["glType"]), img_data);

				break;
			}

			case texture_type::_2d:
			{
				ImmutableTexture2D* p_2d_tex = dynamic_cast<ImmutableTexture2D*>(p_contained_texture);
				if (is_compressed)
					p_2d_tex->setMipmapLevelData(mipmap_level, static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]), imageSize.integer, img_data);
				else
					p_2d_tex->setMipmapLevelData(mipmap_level, static_cast<PixelLayout>(ktx_scalar_preamble["glFormat"]),
					static_cast<PixelDataType>(ktx_scalar_preamble["glType"]), img_data);

				break;
			}

			case texture_type::_3d:
			{
				ImmutableTexture3D* p_3d_tex = dynamic_cast<ImmutableTexture3D*>(p_contained_texture);
				if (is_compressed)
					p_3d_tex->setMipmapLevelData(mipmap_level, static_cast<InternalPixelFormatCompressed>(ktx_scalar_preamble["glInternalFormat"]), imageSize.integer, img_data);
				else
					p_3d_tex->setMipmapLevelData(mipmap_level, static_cast<PixelLayout>(ktx_scalar_preamble["glFormat"]),
					static_cast<PixelDataType>(ktx_scalar_preamble["glType"]), img_data);

				break;
			}

			default:
				set_error_state(true);
				const char* err_msg = "KTX load error: unsupported texture type";
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}

			//Account for mipmap value padding
			char mipmap_padding[3];
			int padding_count = 3 - (imageSize.integer + 3) % 4;
			if (ktx_input_buffer.sgetn(mipmap_padding, padding_count) < padding_count)
			{
				set_error_state(true);
				std::string err_msg = std::string("Invalid mipmap padding value found while reading from file ") +
					file_name + std::string(". File is damaged or is in invalid format");
				set_error_string(err_msg);
				call_error_callback(err_msg);
				return false;
			}
			offset += padding_count;

			delete img_data;
		}
	}

	//Generate mipmap levels if needed
	if (!ktx_scalar_preamble["numberOfMipmapLevels"] && !(tex_type == texture_type::cubemap && ktx_scalar_preamble["numberOfArrayElements"] > 0))
		p_contained_texture->generateMipmapLevels();

	initialization_status = true;

	return true;
}

bool KTXTexture::extractPreambleScalarUint32(std::string parameter_name)
{
	single_word32 uint32_scalar;
	if (ktx_input_buffer.sgetn(uint32_scalar.byte_storage, 4) < 4)
	{
		set_error_state(true);
		std::string err_msg = std::string("Unable to read value of parameter \"") + parameter_name + std::string("\" from the preamble of KTX file (") +
			file_name + std::string("). File is damaged or is in incompatible format");
		set_error_string(err_msg);
		call_error_callback(err_msg);
		return false;
	}

	uint32_t ival = endianness_conversion_required ? swapWord32(uint32_scalar.integer) : uint32_scalar.integer;
	if (!ktx_scalar_preamble.insert(std::make_pair(parameter_name, ival)).second)
		ktx_scalar_preamble[parameter_name] = ival;
	return true;
}

inline KTXTexture::Endianness KTXTexture::getPlatformEndianness() const
{
	single_word32 number = { 0x01020304 };
	if (number.byte_storage[0] == 1)
		return Endianness::BIG_ENDIAN;
	else
		return Endianness::LITTLE_ENDIAN;
}

inline uint32_t KTXTexture::swapWord32(uint32_t word32) const
{
	single_word32 input_word, swapped_word;
	input_word.integer = word32;

	swapped_word.byte_storage[0] = input_word.byte_storage[3];
	swapped_word.byte_storage[1] = input_word.byte_storage[2];
	swapped_word.byte_storage[2] = input_word.byte_storage[1];
	swapped_word.byte_storage[3] = input_word.byte_storage[0];

	return swapped_word.integer;
}

inline uint16_t KTXTexture::swapWord16(uint16_t word16) const
{
	single_word16 input_word, swapped_word;
	input_word.integer = word16;
	swapped_word.byte_storage[0] = input_word.byte_storage[1];
	swapped_word.byte_storage[1] = input_word.byte_storage[0];

	return swapped_word.integer;
}

void* KTXTexture::convertBufferEndianness(void* buffer, size_t buffer_size, MachineWordSize word_size) const
{
	switch (word_size)
	{

	case KTXTexture::MachineWordSize::_16_BIT:
	{
		if (!(buffer_size % 2))
		{
			set_error_state(true);
			const char* err_msg = "Endianness conversion error: buffer size in bytes must be a multiple of machine word size in bytes";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return nullptr;
		}

		uint16_t* _16bit_buffer = reinterpret_cast<uint16_t*>(buffer);
		for (unsigned int i = 0; i < buffer_size / 2; ++i)
			_16bit_buffer[i] = swapWord16(_16bit_buffer[i]);

		break;
	}

	case KTXTexture::MachineWordSize::_32_BIT:
	{
		if (!(buffer_size % 4))
		{
			set_error_state(true);
			const char* err_msg = "Endianness conversion error: buffer size in bytes must be a multiple of machine word size in bytes";
			set_error_string(err_msg);
			call_error_callback(err_msg);
			return nullptr;
		}

		uint32_t* _32bit_buffer = reinterpret_cast<uint32_t*>(buffer);
		for (unsigned int i = 0; i < buffer_size / 4; ++i)
			_32bit_buffer[i] = swapWord32(_32bit_buffer[i]);

		break;
	}

	}

	return buffer;
}

ImmutableTexture* KTXTexture::getContainedTexture() const
{
	if (!initialization_status)
		return nullptr;
	else
		return p_contained_texture;
}

void KTXTexture::releaseTexture()
{
	if (initialization_status)
	{
		delete p_contained_texture;
		p_contained_texture = nullptr;
		initialization_status = false;
		if (ktx_input_buffer.is_open())
			ktx_input_buffer.close();
	}
}

KTXTexture::KeyValueElem::KeyValueElem(size_t elem_size, const void* elem_data)
{
	this->elem_size = elem_size;
	this->elem_data = malloc(elem_size);
	memcpy(this->elem_data, elem_data, elem_size);
}

KTXTexture::KeyValueElem::KeyValueElem(const KeyValueElem& other)
{
	elem_size = other.elem_size;
	elem_data = malloc(elem_size);
	memcpy(elem_data, other.elem_data, elem_size);
}

KTXTexture::KeyValueElem::KeyValueElem(KeyValueElem&& other)
{
	elem_size = other.elem_size;
	elem_data = other.elem_data;

	other.elem_data = nullptr;	//The object, which is getting destroyed receives nullptr and dies without attempt to clear the data that belonged to it. This data is attached to the new object instead.
}

KTXTexture::KeyValueElem::~KeyValueElem()
{
	if (elem_data)
		free(elem_data);
}

bool KTXTexture::KeyValueElem::operator==(const KTXTexture::KeyValueElem& other) const
{
	if (elem_size != other.elem_size)return false;
	return memcmp(elem_data, other.elem_data, elem_size) == 0;
}

bool KTXTexture::KeyValueElem::operator!=(const KTXTexture::KeyValueElem& other) const
{
	return !this->operator==(other);
}

KTXTexture::KeyValueElem& KTXTexture::KeyValueElem::operator=(const KTXTexture::KeyValueElem& other)
{
	elem_size = other.elem_size;
	memcpy(elem_data, other.elem_data, elem_size);
	return *this;
}

KTXTexture::KeyValueElem& KTXTexture::KeyValueElem::operator=(KTXTexture::KeyValueElem&& other)
{
	size_t aux_elem_size = elem_size;
	void *aux_elem_data = elem_data;

	elem_size = other.elem_size;
	elem_data = other.elem_data;

	other.elem_size = aux_elem_size;
	other.elem_data = aux_elem_data;

	return *this;
}

bool KTXTexture::KeyValueElem::operator<(const KeyValueElem& other) const
{
	if (elem_size == other.elem_size)
	{
		return memcmp(elem_data, other.elem_data, elem_size) < 0;
	}
	else return elem_size<other.elem_size;
}

bool KTXTexture::KeyValueElem::operator>(const KeyValueElem& other) const
{
	if (elem_size == other.elem_size)
	{
		return memcmp(elem_data, other.elem_data, elem_size)>0;
	}
	else return elem_size > other.elem_size;
}

bool KTXTexture::KeyValueElem::operator<=(const KeyValueElem& other) const
{
	return !this->operator>(other);
}

bool KTXTexture::KeyValueElem::operator>=(const KeyValueElem& other) const
{
	return !this->operator<(other);
}

size_t KTXTexture::KeyValueElem::size() const{ return elem_size; }

const void* KTXTexture::KeyValueElem::data() const{ return elem_data; }