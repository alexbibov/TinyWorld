#include "TransparentBox.h"
#include "ImageUnit.h"

using namespace tiny_world;


typedef AbstractRenderableObjectExtensionAggregator<AbstractRenderableObjectHDRBloomEx> ExtensionAggregator;


const std::string TransparentBox::rendering_program0_name = "TransparentBox::ray_cast";
const std::string TransparentBox::rendering_program1_name = "TransparentBox::proxy_geometry_light_attenuation";
const std::string TransparentBox::rendering_program2_name = "TransparentBox::proxy_geometry_eye_view_blend_in";


void TransparentBox::applyScreenSize(const uvec2& screen_size)
{
	//Reallocation of the following buffers usually means that the previously allocated ones get destroyed

	//Reallocate the light buffer
	light_buffer_texture = ImmutableTexture2D{ "transparent_box_light_buffer" };
	light_buffer_texture.allocateStorage(1, 1, TextureSize{ screen_size.x, screen_size.y, 0 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

	//Reallocate the eye buffer
	eye_buffer_texture = ImmutableTexture2D{ "transparent_box_eye_buffer" };
	eye_buffer_texture.allocateStorage(1, 1, TextureSize{ screen_size.x, screen_size.y, 0 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

	//Reallocate the bloom texture
	bloom_texture = ImmutableTexture2D{ "transparent_box_bloom" };
	bloom_texture.allocateStorage(1, 1, TextureSize{ screen_size.x, screen_size.y, 0 }, InternalPixelFormat::SIZED_FLOAT_RGBA32);

	//Reallocate the depth buffer
	depth_texture = ImmutableTexture2D{ "transparent_box_depth" };
	depth_texture.allocateStorage(1, 1, TextureSize{ screen_size.x, screen_size.y, 0 }, InternalPixelFormat::SIZED_DEPTH24);

	//Register light and eye buffer textures and bloom texture to the texture pool of the object
	if (!light_buffer_texture_ref_code)
		light_buffer_texture_ref_code = registerTexture(light_buffer_texture);
	else
		updateTexture(light_buffer_texture_ref_code, light_buffer_texture);


	//Update framebuffer objects
	eye_buffer.defineViewport(0, 0, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y));
	eye_buffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &eye_buffer_texture });
	eye_buffer.attachTexture(FramebufferAttachmentPoint::COLOR1, FramebufferAttachmentInfo{ 0, 0, &bloom_texture });
	eye_buffer.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &depth_texture });

	light_buffer.defineViewport(0, 0, static_cast<float>(screen_size.x), static_cast<float>(screen_size.y));
	light_buffer.attachTexture(FramebufferAttachmentPoint::COLOR0, FramebufferAttachmentInfo{ 0, 0, &light_buffer_texture });
	light_buffer.attachTexture(FramebufferAttachmentPoint::DEPTH, FramebufferAttachmentInfo{ 0, 0, &depth_texture });

	//Attach textures to rendering canvas filter
	canvas_filter.defineColorTexture(eye_buffer_texture);
	canvas_filter.defineBloomTexture(bloom_texture);
}


std::pair<bool, vec3> TransparentBox::compute_intersection_point(const vec4& Plane, const vec3& A, const vec3& B)
{
	vec3 N = vec3{ Plane.x, Plane.y, Plane.z };
	float D = Plane.w;

	vec3 v3S = A;
	vec3 v3R = B - A;

	float t;
	const float Inf = std::numeric_limits<float>::infinity();

	t = std::min((-D - N.dot_product(v3S)) / N.dot_product(v3R), Inf);

	return t >= 0.0f && t <= 1.0f ? std::make_pair(true, v3S + t*v3R) : std::make_pair(false, vec3{ Inf });
}


void TransparentBox::generate_proxy_geometry()
{
	//Store information about number of primary samples that was actual during previous evaluation of this function
	//This is needed to update memory allocations only when it becomes necessary
	if (num_primary_samples_old != num_primary_samples)
	{
		num_primary_samples_old = static_cast<int>(num_primary_samples);

		//Reallocate vertex buffer memory
		glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object1);
		glBufferData(GL_ARRAY_BUFFER,
			(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity()) * 8 * num_primary_samples,
			NULL, GL_DYNAMIC_DRAW);
		glBindVertexArray(ogl_vertex_attribute_object1);
		glBindVertexBuffer(0, ogl_vertex_buffer_object1, 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity());

		//Reallocate buffer of vertex binding offsets
		if (p_vertex_binding_offsets) delete[] p_vertex_binding_offsets;
		p_vertex_binding_offsets = new uint32_t[num_primary_samples];

		//Reallocate buffer containing for each slice number of generic vertices that form this slice
		if (p_num_slice_vertices) delete[] p_num_slice_vertices;
		p_num_slice_vertices = new uint32_t[num_primary_samples];
	}	


	//Compute common normal vector of the cutting planes that form proxy geometry
	vec3 v3CuttingPlaneNormal = v3AverageLightDirection.dot_product(v3ViewerDirection) >= 0 ?
		(v3AverageLightDirection.get_normalized() + v3ViewerDirection.get_normalized()) :
		(v3AverageLightDirection.get_normalized() - v3ViewerDirection.get_normalized());
	mat4 m4WorldToObjectRotation = getObjectTransform().inverse();
	m4WorldToObjectRotation[0][3] = 0;
	m4WorldToObjectRotation[1][3] = 0;
	m4WorldToObjectRotation[2][3] = 0;
	vec4 v4Aux = m4WorldToObjectRotation * vec4{ v3CuttingPlaneNormal, 1.0f };
	v3CuttingPlaneNormal = (vec3{ v4Aux.x, v4Aux.y, v4Aux.z } / v4Aux.w).get_normalized();


	//Compute offset value for each cutting plane
	vec3 v3BoxScale = getObjectScale();
	std::array<vec3, 8> proxy_cube_vertices = {
		vec3{ -width / 2 * v3BoxScale.x, -height / 2 * v3BoxScale.y, depth / 2 * v3BoxScale.z },
		vec3{ width / 2 * v3BoxScale.x, -height / 2 * v3BoxScale.y, depth / 2 * v3BoxScale.z },
		vec3{ width / 2 * v3BoxScale.x, -height / 2 * v3BoxScale.y, -depth / 2 * v3BoxScale.z },
		vec3{ -width / 2 * v3BoxScale.x, -height / 2 * v3BoxScale.y, -depth / 2 * v3BoxScale.z },

		vec3{ -width / 2 * v3BoxScale.x, height / 2 * v3BoxScale.y, depth / 2 * v3BoxScale.z },
		vec3{ width / 2 * v3BoxScale.x, height / 2 * v3BoxScale.y, depth / 2 * v3BoxScale.z },
		vec3{ width / 2 * v3BoxScale.x, height / 2 * v3BoxScale.y, -depth / 2 * v3BoxScale.z },
		vec3{ -width / 2 * v3BoxScale.x, height / 2 * v3BoxScale.y, -depth / 2 * v3BoxScale.z } };

	std::pair<std::array<vec3, 8>::const_iterator, std::array<vec3, 8>::const_iterator> minmax_vertices =
		std::minmax_element(proxy_cube_vertices.begin(), proxy_cube_vertices.end(),
		[&v3CuttingPlaneNormal](vec3 elem1, vec3 elem2) -> bool 
	{
		return v3CuttingPlaneNormal.dot_product(elem1 - elem2) < 0;
	});
	float min_offset = (*minmax_vertices.first).dot_product(v3CuttingPlaneNormal);
	float max_offset = (*minmax_vertices.second).dot_product(v3CuttingPlaneNormal);
	float delta = (max_offset - min_offset) / (num_primary_samples - 1.0f);
	min_offset += delta / 3.0f; max_offset -= delta / 3.0f;
	delta = (max_offset - min_offset) / (num_primary_samples - 1.0f);


	//Compute intersections between the proxy cube and each cutting plane
	uint32_t current_vertex_binding_offset = 0;
	for (int i = 0; i < static_cast<int>(num_primary_samples); ++i)
	{
		float D = -min_offset - delta * i;
		
		//The maximal number of intersections between a cutting plane and the proxy box can not exceed 6
		std::array<vec3, 6> intersection_points;
		uint32_t num_intersections = 0;
		std::pair<bool, vec3> intersection;


		//Compute intersection points between the cutting plane and each of the edges of the box
		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[0], proxy_cube_vertices[4])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[1], proxy_cube_vertices[5])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[2], proxy_cube_vertices[6])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[3], proxy_cube_vertices[7])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}


		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[0], proxy_cube_vertices[1])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[1], proxy_cube_vertices[2])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[2], proxy_cube_vertices[3])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[3], proxy_cube_vertices[0])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}
		

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[4], proxy_cube_vertices[5])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[5], proxy_cube_vertices[6])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[6], proxy_cube_vertices[7])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}

		if ((intersection = compute_intersection_point(vec4{ v3CuttingPlaneNormal, D }, proxy_cube_vertices[7], proxy_cube_vertices[4])).first)
		{
			intersection_points[num_intersections] = intersection.second;
			++num_intersections;
		}


		//Find the center of mass of the intersection points
		vec3 v3MassCenter{ 0.0f };
		std::for_each(intersection_points.begin(), intersection_points.begin() + num_intersections,
			[&v3MassCenter](const vec3& element) -> void { v3MassCenter = v3MassCenter + element; });
		v3MassCenter /= num_intersections;

		//Sort vertices of the intersection rectangle in counter-clockwise order
		v4Aux = m4WorldToObjectRotation * vec4{ v3ViewerDirection, 1.0f };
		vec3 v3ViewerDirectionInScaledObjectSpace = vec3{ v4Aux.x, v4Aux.y, v4Aux.z } / v4Aux.w;

		v4Aux = m4WorldToObjectRotation * vec4{ v3ViewerUpVector, 1.0f };
		vec3 v3ViewerUpVectorInScaledObjectSpace = vec3{ v4Aux.x, v4Aux.y, v4Aux.z } / v4Aux.w;

		vec3 v3ViewerSideVectorInScaledObjectSpace = v3ViewerUpVectorInScaledObjectSpace.cross_product(-v3ViewerDirectionInScaledObjectSpace);

		//Project the mass center and the intersection points of the currently processed cross-section onto the XY-plane of the observer
		std::sort(intersection_points.begin(), intersection_points.begin() + num_intersections, 
			[&v3MassCenter, &v3ViewerSideVectorInScaledObjectSpace, &v3ViewerUpVectorInScaledObjectSpace](const vec3& elem1, const vec3& elem2) -> bool
		{
			vec3 e1 = elem1 - v3MassCenter, e2 = elem2 - v3MassCenter;
			vec2 elem1_projected{ e1.dot_product(v3ViewerSideVectorInScaledObjectSpace), e1.dot_product(v3ViewerUpVectorInScaledObjectSpace) };
			vec2 elem2_projected{ e2.dot_product(v3ViewerSideVectorInScaledObjectSpace), e2.dot_product(v3ViewerUpVectorInScaledObjectSpace) };

			float tan_value1 = elem1_projected.y / elem1_projected.x;
			float sign_value1 = elem1_projected.x >= 0 ? 1.0f : -1.0f;

			float tan_value2 = elem2_projected.y / elem2_projected.x;
			float sign_value2 = elem2_projected.x >= 0 ? 1.0f : -1.0f;

			if (sign_value1 != sign_value2) 
				return sign_value1 > 0 && sign_value2 < 0;
			else
				return tan_value1 < tan_value2;
		});


		//Generate texture coordinates
		std::array<vec3, 7> texture_coordinates;
		texture_coordinates[0].x = v3MassCenter.x / (v3BoxScale.x * width) + 0.5f;
		texture_coordinates[0].y = v3MassCenter.y / (v3BoxScale.y * height) + 0.5f;
		texture_coordinates[0].z = v3MassCenter.z / (v3BoxScale.z * depth) + 0.5f;
		for (int j = 1; j < static_cast<int>(num_intersections)+1; ++j)
		{
			texture_coordinates[j].x = intersection_points[j - 1].x / (v3BoxScale.x * width) + 0.5f;
			texture_coordinates[j].y = intersection_points[j - 1].y / (v3BoxScale.y * height) + 0.5f;
			texture_coordinates[j].z = intersection_points[j - 1].z / (v3BoxScale.z * depth) + 0.5f;
		}

		//Populate vertex buffer with data
		void* cutting_plane_data = malloc((vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity()) * (num_intersections + 2));
		memcpy(cutting_plane_data, vec4{ v3MassCenter.x / v3BoxScale.x, v3MassCenter.y / v3BoxScale.y, v3MassCenter.z / v3BoxScale.z, 1.0f }.getDataAsArray(), sizeof(vec4::value_type)*vec4::dimension);
		memcpy(static_cast<char*>(cutting_plane_data)+sizeof(vec4::value_type)*vec4::dimension, texture_coordinates[0].getDataAsArray(), sizeof(vec3::value_type)*vec3::dimension);
		for (int j = 0; j < static_cast<int>(num_intersections) + 1; ++j)
		{
			v4Aux = vec4{ intersection_points[j == num_intersections ? 0 : j], 1.0f };
			v4Aux.x /= v3BoxScale.x; v4Aux.y /= v3BoxScale.y; v4Aux.z /= v3BoxScale.z;
			memcpy(static_cast<char*>(cutting_plane_data)+(sizeof(vec4::value_type)*vec4::dimension + sizeof(vec3::value_type)*vec3::dimension)*(j + 1),
				v4Aux.getDataAsArray(), sizeof(vec4::value_type)*vec4::dimension);
			memcpy(static_cast<char*>(cutting_plane_data)+(sizeof(vec4::value_type)*vec4::dimension + sizeof(vec3::value_type)*vec3::dimension)*(j + 1) + sizeof(vec4::value_type)*vec4::dimension,
				texture_coordinates[(j == num_intersections ? 0 : j) + 1].getDataAsArray(), sizeof(vec3::value_type)*vec3::dimension);
		}

		glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object1);
		glBufferSubData(GL_ARRAY_BUFFER, static_cast<GLintptr>(current_vertex_binding_offset),
			(vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity()) * (num_intersections + 2),
			cutting_plane_data);
		free(cutting_plane_data);

		p_vertex_binding_offsets[i] = current_vertex_binding_offset;
		current_vertex_binding_offset += (vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity())*(num_intersections + 2);

		p_num_slice_vertices[i] = num_intersections + 2;
	}
}


void TransparentBox::update_vertex_data()
{
	char *vertex_data_buffer = new char[vertex_attribute_position::getCapacity() * 8];

	vertex_attribute_position::value_type* p_vertex =
		reinterpret_cast<vertex_attribute_position::value_type*>(vertex_data_buffer);

	//Lower-left-near corner
	p_vertex[0] = -width / 2; p_vertex[1] = -height / 2; p_vertex[2] = depth / 2; p_vertex[3] = 1.0f;

	//Lower-right-near corner
	p_vertex[4] = width / 2; p_vertex[5] = -height / 2; p_vertex[6] = depth / 2; p_vertex[7] = 1.0f;

	//Lower-right-far corner
	p_vertex[8] = width / 2; p_vertex[9] = -height / 2; p_vertex[10] = -depth / 2; p_vertex[11] = 1.0f;

	//Lower-left-far corner
	p_vertex[12] = -width / 2; p_vertex[13] = -height / 2; p_vertex[14] = -depth / 2; p_vertex[15] = 1.0f;


	//Upper-left-near corner
	p_vertex[16] = -width / 2; p_vertex[17] = height / 2; p_vertex[18] = depth / 2; p_vertex[19] = 1.0f;

	//Upper-right-near corner
	p_vertex[20] = width / 2; p_vertex[21] = height / 2; p_vertex[22] = depth / 2; p_vertex[23] = 1.0f;

	//Upper-right-far corner
	p_vertex[24] = width / 2; p_vertex[25] = height / 2; p_vertex[26] = -depth / 2; p_vertex[27] = 1.0f;

	//Upper-left-far corner
	p_vertex[28] = -width / 2; p_vertex[29] = height / 2; p_vertex[30] = -depth / 2; p_vertex[31] = 1.0f;


	glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object0);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_attribute_position::getCapacity() * 8, vertex_data_buffer);


	delete[] vertex_data_buffer;
}


void TransparentBox::init_transparent_box()
{
	//Switch default rendering mode
	selectRenderingMode(TW_RENDERING_MODE_RAY_CAST_GAS);

	//Firstly, create OpenGL vertex attribute objects, and OpenGL vertex and index buffer objects
	GLuint VAOs[2];
	glGenVertexArrays(2, VAOs);
	ogl_vertex_attribute_object0 = VAOs[0];
	ogl_vertex_attribute_object1 = VAOs[1];
	glBindVertexArray(ogl_vertex_attribute_object0);

	GLuint buf_objs[3];
	glGenBuffers(3, buf_objs);
	ogl_vertex_buffer_object0 = buf_objs[0];
	ogl_index_buffer_object0 = buf_objs[1];
	ogl_vertex_buffer_object1 = buf_objs[2];


	//Setup vertex attribute object and vertex buffer object used by the rendering modes based on ray casting 
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

	//Allocate vertex buffer
	glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object0);
	glBufferData(GL_ARRAY_BUFFER, vertex_attribute_position::getCapacity() * 8, NULL, GL_STATIC_DRAW);


	//Generate vertex data based on the current settings for transparent box's dimensions
	update_vertex_data();
	glBindVertexBuffer(0, ogl_vertex_buffer_object0, 0, vertex_attribute_position::getCapacity());

	//Specify index data (NOTE: front face ordering is  v3, v4, v2, v1; back face ordering is v3, v2, v4, v1)
	GLushort index_data_buffer[29] = { 2, 1, 3, 0, 0xFFFF,
		6, 7, 5, 4, 0xFFFF,
		5, 4, 1, 0, 0xFFFF,
		6, 2, 7, 3, 0xFFFF,
		4, 7, 0, 3, 0xFFFF,
		6, 5, 2, 1
	};
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_index_buffer_object0);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 29, index_data_buffer, GL_STATIC_DRAW);


	//Setup vertex attribute object used by the rendering modes based on proxy geometry
	glBindVertexArray(ogl_vertex_attribute_object1);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);
	
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

	glEnableVertexAttribArray(vertex_attribute_texcoord_3d::getId());
	vertex_attribute_texcoord_3d::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

	//Attach vertex buffer containing proxy geometry to the corresponding vertex attribute object
	glBindVertexBuffer(0, ogl_vertex_buffer_object1, 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity());


	//Setup medium texture sampler
	medium_sampler_ref_code = createTextureSampler("TransparentBox::medium_sampler", SamplerMagnificationFilter::LINEAR, SamplerMinificationFilter::LINEAR_MIPMAP_LINEAR,
		SamplerWrapping{ SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE, SamplerWrappingMode::CLAMP_TO_EDGE });
	setTextureUnitOffset(0);


	//Setup rendering programs
	if (!rendering_program0_ref_code)
	{
		rendering_program0_ref_code = createCompleteShaderProgram(rendering_program0_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

		//Initialize rendering program
		Shader vertex_shader{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_RT.vp.glsl", ShaderType::VERTEX_SHADER, "TransparentBox_RT_VP" };
		Shader fragment_program{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_RT.fp.glsl", ShaderType::FRAGMENT_SHADER, "TransparentBox_RT_FP" };

		retrieveShaderProgram(rendering_program0_ref_code)->addShader(vertex_shader);
		retrieveShaderProgram(rendering_program0_ref_code)->addShader(fragment_program);

		retrieveShaderProgram(rendering_program0_ref_code)->bindVertexAttributeId("vertex_position", vertex_attribute_position::getId());

		retrieveShaderProgram(rendering_program0_ref_code)->link();
	}

	if (!(rendering_program1_ref_code || rendering_program2_ref_code))
	{
		rendering_program1_ref_code = createCompleteShaderProgram(rendering_program1_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });
		rendering_program2_ref_code = createCompleteShaderProgram(rendering_program2_name, { PipelineStage::VERTEX_SHADER, PipelineStage::FRAGMENT_SHADER });

		//Initialize volume rendering shaders
		Shader PG1_VP{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_PG1.vp.glsl", ShaderType::VERTEX_SHADER, "TransparentBox_PG1_VP" };
		Shader PG1_FP{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_PG1.fp.glsl", ShaderType::FRAGMENT_SHADER, "TransparentBox_PG1_FP" };
		Shader PG2_VP{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_PG2.vp.glsl", ShaderType::VERTEX_SHADER, "TransparentBox_PG2_VP" };
		Shader PG2_FP{ ShaderProgram::getShaderBaseCatalog() + "TransparentBox_PG2.fp.glsl", ShaderType::FRAGMENT_SHADER, "TransparentBox_PG2_FP" };

		retrieveShaderProgram(rendering_program1_ref_code)->addShader(PG1_VP);
		retrieveShaderProgram(rendering_program1_ref_code)->addShader(PG1_FP);
		retrieveShaderProgram(rendering_program2_ref_code)->addShader(PG2_VP);
		retrieveShaderProgram(rendering_program2_ref_code)->addShader(PG2_FP);

		retrieveShaderProgram(rendering_program1_ref_code)->bindVertexAttributeId("v4VertexPosition", vertex_attribute_position::getId());
		retrieveShaderProgram(rendering_program1_ref_code)->bindVertexAttributeId("v3TextureCoordinate3D", vertex_attribute_texcoord_3d::getId());
		retrieveShaderProgram(rendering_program2_ref_code)->bindVertexAttributeId("v4VertexPosition", vertex_attribute_position::getId());
		retrieveShaderProgram(rendering_program2_ref_code)->bindVertexAttributeId("v3TextureCoordinate3D", vertex_attribute_texcoord_3d::getId());
		
		retrieveShaderProgram(rendering_program1_ref_code)->link();
		retrieveShaderProgram(rendering_program2_ref_code)->link();
	}


	//Setup light and eye buffers
	light_buffer.setStringName("transparent_box_light_buffer");
	light_buffer.setClearDepth(1.0);
	light_buffer.setBlendEquation(ColorBlendEquation::ADD);
	light_buffer.setColorBlendEnableState(true);
	light_buffer.setDepthBufferUpdateFlag(false);
	light_buffer.setCullTestEnableState(false);
	light_buffer_texture.setStringName("transparent_box_light_buffer_texture");

	eye_buffer.setStringName("transparent_box_eye_buffer");
	eye_buffer.setClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	eye_buffer.setClearDepth(1.0);
	eye_buffer.setBlendEquation(ColorBlendEquation::ADD);
	eye_buffer.setColorBlendEnableState(true);
	eye_buffer.setDepthBufferUpdateFlag(false);
	eye_buffer_texture.setStringName("transparent_box_eye_buffer_texture");
	bloom_texture.setStringName("transparent_box_eye_buffer_bloom_texture");


	//Set canvas filter parameters
	canvas_filter.setBloomImpact(0.2f);
}


void TransparentBox::free_ogl_resources()
{
	//Release OpenGL resources owned by the object
	GLuint VAOs[2];
	VAOs[0] = ogl_vertex_attribute_object0;
	VAOs[1] = ogl_vertex_attribute_object1;

	GLuint VBOs[3];
	VBOs[0] = ogl_vertex_buffer_object0;
	VBOs[1] = ogl_index_buffer_object0;
	VBOs[2] = ogl_vertex_buffer_object1;

	glDeleteVertexArrays(2, VAOs);
	glDeleteBuffers(3, VBOs);


	//Release memory allocations owned by the object
	if (p_vertex_binding_offsets)
	{
		delete[] p_vertex_binding_offsets;
		p_vertex_binding_offsets = nullptr;
	}

	if (p_num_slice_vertices)
	{
		delete[] p_num_slice_vertices;
		p_num_slice_vertices = nullptr;
	}
	

	//Update reference counter
	(*p_ref_counter)--;

	//Check state of the reference counter and release shared resources if counter's value is zero
	if (!(*p_ref_counter))
	{
		delete p_ref_counter;
		glDeleteBuffers(1, &ogl_index_buffer_object0);
	}
}


TransparentBox::TransparentBox(float width /* = 1.0f */, float height /* = 1.0f */, float depth /* = 1.0f */) :
AbstractRenderableObject("TransparentBox"),
width{ width }, height{ height }, depth{ depth }, v3AverageLightDirection{ 0.0f }, v3CumulativeLightColor{ 0.0f }, 
num_primary_samples_old{ -1 }, num_primary_samples{ 15 }, num_secondary_samples{ 2 }, solid_angle{ 0.01f },
p_vertex_binding_offsets{ nullptr }, p_num_slice_vertices{ nullptr }, should_use_rgb_channel{ false }, medium_color{ 1.0f }, 
should_use_colormap{ false }, current_rendering_pass{ 0 }, p_renderer_projection{ nullptr }, p_render_target{ nullptr }
{
	//Initialize reference counter
	p_ref_counter = new uint32_t{ 1 };

	//Perform the rest of initialization particulars
	init_transparent_box();
}


TransparentBox::TransparentBox(std::string transparent_box_string_name, float width /* = 1.0f */, float height /* = 1.0f */, float depth /* = 1.0f */) :
AbstractRenderableObject("TransparentBox", transparent_box_string_name),
width{ width }, height{ height }, depth{ depth }, v3AverageLightDirection{ 0.0f }, v3CumulativeLightColor{ 0.0f },
num_primary_samples_old{ -1 }, num_primary_samples{ 15 }, num_secondary_samples{ 2 }, solid_angle{ 0.01f },
p_vertex_binding_offsets{ nullptr }, p_num_slice_vertices{ nullptr }, should_use_rgb_channel{ false }, medium_color{ 1.0f }, 
should_use_colormap{ false }, current_rendering_pass{ 0 }, p_renderer_projection{ nullptr }, p_render_target{ nullptr }
{
	//Initialize reference counter
	p_ref_counter = new uint32_t{ 1 };

	//Perform the rest of initialization particulars
	init_transparent_box();
}


TransparentBox::TransparentBox(const TransparentBox& other) : 
AbstractRenderableObject(other), 
AbstractRenderableObjectTextured(other),
ExtensionAggregator(other), 
width{ other.width }, height{ other.height }, depth{ other.depth }, 
light_source_directions(other.light_source_directions), 
v3AverageLightDirection{ other.v3AverageLightDirection }, v3CumulativeLightColor{other.v3CumulativeLightColor},
num_primary_samples_old{ -1 }, num_primary_samples{ other.num_primary_samples }, num_secondary_samples{ other.num_secondary_samples },
solid_angle{ other.solid_angle }, p_vertex_binding_offsets{ nullptr }, p_num_slice_vertices{ nullptr }, 
ogl_index_buffer_object0{ other.ogl_index_buffer_object0 }, 
medium_sampler_ref_code{ other.medium_sampler_ref_code }, should_use_rgb_channel{ other.should_use_rgb_channel }, 
medium_color{ other.medium_color }, should_use_colormap{ other.should_use_colormap }, 
medium_texture_ref_code{ other.medium_texture_ref_code }, colormap_texture_ref_code{ other.colormap_texture_ref_code }, 
light_buffer_texture{ other.light_buffer_texture }, light_buffer_texture_ref_code{ other.light_buffer_texture_ref_code },
eye_buffer_texture{ other.eye_buffer_texture },
bloom_texture{ other.bloom_texture },
depth_texture{ other.depth_texture },
eye_buffer{ other.eye_buffer }, light_buffer{ other.light_buffer },
rendering_program0_ref_code{ other.rendering_program0_ref_code }, 
rendering_program1_ref_code{ other.rendering_program1_ref_code }, 
rendering_program2_ref_code{ other.rendering_program2_ref_code }, 
current_rendering_pass{ other.current_rendering_pass }, 
lightview_projection{ other.lightview_projection }, p_renderer_projection{ other.p_renderer_projection }, 
p_render_target{ other.p_render_target }, p_ref_counter{ other.p_ref_counter }
{
	//Update reference counter
	(*p_ref_counter)++;

	//Create and initialize vertex attribute objects and vertex buffer objects
	GLuint VAOs[2];
	glGenVertexArrays(2, VAOs);
	ogl_vertex_attribute_object0 = VAOs[0];
	ogl_vertex_attribute_object1 = VAOs[1];

	GLuint VBOs[2];
	glGenBuffers(2, VBOs);
	ogl_vertex_buffer_object0 = VBOs[0];
	ogl_vertex_buffer_object1 = VBOs[1];

	//Setup vertex attribute object, which is used by rendering modes implementing ray casting algorithms
	glBindVertexArray(ogl_vertex_attribute_object0);
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

	//Setup vertex buffer implementing proxy bounding box
	glBindBuffer(GL_ARRAY_BUFFER, ogl_vertex_buffer_object0);
	glBufferData(GL_ARRAY_BUFFER, vertex_attribute_position::getCapacity() * 8, NULL, GL_STATIC_DRAW);

	//Populate vertex buffer containing proxy bounding box with data
	update_vertex_data();

	//Attach vertex buffer to vertex attribute object
	glBindVertexBuffer(0, ogl_vertex_buffer_object0, 0, vertex_attribute_position::getCapacity());

	//Attach index buffer to vertex attribute object
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ogl_index_buffer_object0);


	//Setup vertex attribute object, which is used by rendering modes implementing proxy geometry rendering schemes
	glBindVertexArray(ogl_vertex_attribute_object1);
	glEnableVertexAttribArray(vertex_attribute_position::getId());
	vertex_attribute_position::setVertexAttributeBufferLayout(0, 0);

	glEnableVertexAttribArray(vertex_attribute_texcoord_3d::getId());
	vertex_attribute_texcoord_3d::setVertexAttributeBufferLayout(vertex_attribute_position::getCapacity(), 0);

	//Attach vertex buffer containing proxy geometry data to the corresponding vertex attribute object
	glBindVertexBuffer(0, ogl_vertex_buffer_object1, 0, vertex_attribute_position::getCapacity() + vertex_attribute_texcoord_3d::getCapacity());
}


TransparentBox::TransparentBox(TransparentBox&& other) : 
AbstractRenderableObject(std::move(other)),
AbstractRenderableObjectTextured(std::move(other)),
ExtensionAggregator(std::move(other)),
width{ other.width }, height{ other.height }, depth{ other.depth },
light_source_directions(std::move(light_source_directions)), 
v3AverageLightDirection{ std::move(other.v3AverageLightDirection) }, v3CumulativeLightColor{ std::move(other.v3CumulativeLightColor) }, 
num_primary_samples_old{ other.num_primary_samples_old }, num_primary_samples{ other.num_primary_samples }, num_secondary_samples{ other.num_secondary_samples },
solid_angle{ other.solid_angle }, p_vertex_binding_offsets{ other.p_vertex_binding_offsets }, p_num_slice_vertices{ other.p_num_slice_vertices }, 
ogl_vertex_attribute_object0{ other.ogl_vertex_attribute_object0 }, 
ogl_vertex_attribute_object1{ other.ogl_vertex_attribute_object1 },
ogl_vertex_buffer_object0{ other.ogl_vertex_buffer_object0 },
ogl_index_buffer_object0{ other.ogl_index_buffer_object0 },
ogl_vertex_buffer_object1{ other.ogl_vertex_buffer_object1 }, 
medium_sampler_ref_code{ std::move(other.medium_sampler_ref_code) }, should_use_rgb_channel{ other.should_use_rgb_channel }, 
medium_color{ std::move(other.medium_color) }, should_use_colormap{ other.should_use_colormap },
medium_texture_ref_code{ std::move(other.medium_texture_ref_code) }, colormap_texture_ref_code{ std::move(other.colormap_texture_ref_code) }, 
light_buffer_texture{ std::move(other.light_buffer_texture) }, light_buffer_texture_ref_code{ std::move(other.light_buffer_texture_ref_code) },
eye_buffer_texture{ std::move(other.eye_buffer_texture) },
bloom_texture{ std::move(other.bloom_texture) }, 
depth_texture{ std::move(other.depth_texture) },
eye_buffer{ std::move(other.eye_buffer) }, light_buffer{ std::move(other.light_buffer) },
rendering_program0_ref_code{ std::move(other.rendering_program0_ref_code) },
rendering_program1_ref_code{ std::move(other.rendering_program1_ref_code) },
rendering_program2_ref_code{ std::move(other.rendering_program2_ref_code) }, 
current_rendering_pass{ other.current_rendering_pass }, 
lightview_projection{ std::move(other.lightview_projection) }, p_renderer_projection{ other.p_renderer_projection }, 
p_render_target{ other.p_render_target }, p_ref_counter{ other.p_ref_counter }
{
	//Update reference counter
	(*p_ref_counter)++;

	//Ensure that no resources that have been moved to the newly created object will get destroyed by destructor of the object being moved
	other.p_vertex_binding_offsets = nullptr;
	other.p_num_slice_vertices = nullptr;
}


TransparentBox::~TransparentBox()
{
	free_ogl_resources();
}


TransparentBox& TransparentBox::operator=(const TransparentBox& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Assign the base classes 
	AbstractRenderableObject::operator=(other);
	AbstractRenderableObjectTextured::operator=(other);
	ExtensionAggregator::operator=(other);

	//Update reference counter
	(*other.p_ref_counter)++;
	(*p_ref_counter)--;

	//Check state of the reference counter of "this" object and destroy shared resources owned by the object, if the object is the last in its sharing class
	if (!(*p_ref_counter))
	{
		delete p_ref_counter;
		glDeleteBuffers(1, &ogl_index_buffer_object0);
	}

	//Copy state of the object
	width = other.width;
	height = other.height;
	depth = other.depth;
	light_source_directions = other.light_source_directions;
	v3AverageLightDirection = other.v3AverageLightDirection;
	v3CumulativeLightColor = other.v3CumulativeLightColor;
	//num_primary_samples_old is left having its old value, so that the OpenGL buffers would be reallocated only when actual need comes
	num_primary_samples = other.num_primary_samples;
	num_secondary_samples = other.num_secondary_samples;
	solid_angle = other.solid_angle;

	//p_vertex_binding_offsets gets reallocated by generate_proxy_geometry() if other.num_primary_samples != num_primary_samples_old
	//p_num_slice_vertices gets reallocated by generate_proxy_geometry() if other.num_primary_samples != num_primary_samples_old

	ogl_index_buffer_object0 = other.ogl_index_buffer_object0;

	medium_sampler_ref_code = other.medium_sampler_ref_code;

	should_use_rgb_channel = other.should_use_rgb_channel;
	medium_color = other.medium_color;
	should_use_colormap = other.should_use_colormap;

	medium_texture_ref_code = other.medium_texture_ref_code;
	colormap_texture_ref_code = other.colormap_texture_ref_code;
	light_buffer_texture = other.light_buffer_texture;
	light_buffer_texture_ref_code = other.light_buffer_texture_ref_code;
	eye_buffer_texture = other.eye_buffer_texture;
	bloom_texture = other.bloom_texture;
	depth_texture = other.depth_texture;
	eye_buffer = other.eye_buffer;
	light_buffer = other.light_buffer;

	rendering_program0_ref_code = other.rendering_program0_ref_code;
	rendering_program1_ref_code = other.rendering_program1_ref_code;
	rendering_program2_ref_code = other.rendering_program2_ref_code;

	current_rendering_pass = other.current_rendering_pass;
	lightview_projection = other.lightview_projection;
	p_renderer_projection = other.p_renderer_projection;
	p_render_target = other.p_render_target;

	p_ref_counter = other.p_ref_counter;

	return *this;
}


TransparentBox& TransparentBox::operator=(TransparentBox&& other)
{
	//Account for the special case of "assignment to itself"
	if (this == &other)
		return *this;

	//Move the base classes
	AbstractRenderableObject::operator=(std::move(other));
	AbstractRenderableObjectTextured::operator=(std::move(other));
	ExtensionAggregator::operator=(std::move(other));

	//Update reference counter
	(*other.p_ref_counter)++;
	(*p_ref_counter)--;

	//Check the updated state of the reference counter and destroy shared resources owned by the object if the object is the last 
	//representative remaining from its sharing class
	if (!(*p_ref_counter))
	{
		delete p_ref_counter;
		glDeleteBuffers(1, &ogl_index_buffer_object0);
	}


	//Move state of the object
	width = other.width;
	height = other.height;
	depth = other.depth;
	light_source_directions = std::move(other.light_source_directions);
	v3AverageLightDirection = std::move(other.v3AverageLightDirection);
	v3CumulativeLightColor = std::move(other.v3CumulativeLightColor);

	num_primary_samples_old = other.num_primary_samples_old;
	num_primary_samples = other.num_primary_samples;
	num_secondary_samples = other.num_secondary_samples;
	solid_angle = other.solid_angle;

	std::swap(p_vertex_binding_offsets, other.p_vertex_binding_offsets);
	std::swap(p_num_slice_vertices, other.p_num_slice_vertices);

	std::swap(ogl_vertex_attribute_object0, other.ogl_vertex_attribute_object0);
	std::swap(ogl_vertex_buffer_object0, other.ogl_vertex_buffer_object0);
	std::swap(ogl_index_buffer_object0, other.ogl_index_buffer_object0);
	std::swap(ogl_vertex_attribute_object1, other.ogl_vertex_attribute_object1);
	std::swap(ogl_vertex_buffer_object1, other.ogl_vertex_buffer_object1);

	medium_sampler_ref_code = std::move(other.medium_sampler_ref_code);

	should_use_rgb_channel = other.should_use_rgb_channel;
	medium_color = std::move(other.medium_color);
	should_use_colormap = other.should_use_colormap;

	medium_texture_ref_code = std::move(other.medium_texture_ref_code);
	colormap_texture_ref_code = std::move(other.colormap_texture_ref_code);
	light_buffer_texture = std::move(other.light_buffer_texture);
	light_buffer_texture_ref_code = std::move(other.light_buffer_texture_ref_code);
	eye_buffer_texture = std::move(other.eye_buffer_texture);
	bloom_texture = std::move(other.bloom_texture);
	depth_texture = std::move(other.depth_texture);
	eye_buffer = std::move(other.eye_buffer);
	light_buffer = std::move(other.light_buffer);
	
	rendering_program0_ref_code = std::move(other.rendering_program0_ref_code);
	rendering_program1_ref_code = std::move(other.rendering_program1_ref_code);
	rendering_program2_ref_code = std::move(other.rendering_program2_ref_code);

	current_rendering_pass = other.current_rendering_pass;
	lightview_projection = std::move(other.lightview_projection);
	p_renderer_projection = other.p_renderer_projection;
	p_render_target = other.p_render_target;

	p_ref_counter = other.p_ref_counter;

	return *this;
}


void TransparentBox::setDimensions(float width, float height, float depth)
{
	this->width = width; this->height = height; this->depth = depth;
	update_vertex_data();
}


void TransparentBox::setDimensions(const vec3& new_dimensions)
{
	width = new_dimensions.x; height = new_dimensions.y; depth = new_dimensions.z;
	update_vertex_data();
}


bool TransparentBox::addLightSourceDirection(const DirectionalLight& directional_light)
{
	//If light object being added is not currently in the list add it, otherwise do nothing and return 'false'
	if (std::find_if(light_source_directions.begin(), light_source_directions.end(),
		[this, &directional_light](const DirectionalLight* light) -> bool
	{
		return light->getId() == directional_light.getId();
	}) != light_source_directions.end())
		return false;
	else
	{
		v3AverageLightDirection *= light_source_directions.size();
		light_source_directions.push_back(&directional_light);
		v3AverageLightDirection = (v3AverageLightDirection + directional_light.getDirection()) / light_source_directions.size();
		v3CumulativeLightColor += directional_light.getColor();
		light_buffer.setClearColor(vec4{ v3CumulativeLightColor, 0.0f });
		return true;
	}
}


bool TransparentBox::removeLightSourceDirection(uint32_t directional_light_id)
{
	//If light object with requested Id is in the list remove it

	std::list<const DirectionalLight*>::const_iterator light_to_remove_position;
	if ((light_to_remove_position = std::find_if(light_source_directions.begin(), light_source_directions.end(),
		[this, directional_light_id](const DirectionalLight* light) -> bool
	{
		return light->getId() == directional_light_id;
	})) != light_source_directions.end())
	{
		v3AverageLightDirection *= light_source_directions.size();
		light_source_directions.erase(light_to_remove_position);
		v3AverageLightDirection = (v3AverageLightDirection - (*light_to_remove_position)->getDirection()) / light_source_directions.size();
		v3CumulativeLightColor -= (*light_to_remove_position)->getColor();
		light_buffer.setClearColor(vec4{ v3CumulativeLightColor, 0.0f });
		return true;
	}

	return false;
}


bool TransparentBox::removeLightSourceDirection(std::string directional_light_string_name)
{
	//Look for the first light object in the list having requested string name. If such object is found, remove it and return 'true'.
	//Otherwise, do nothing and return 'false'

	std::list<const DirectionalLight*>::const_iterator light_to_remove_position;
	if ((light_to_remove_position = std::find_if(light_source_directions.begin(), light_source_directions.end(),
		[this, &directional_light_string_name](const DirectionalLight* light) -> bool
	{
		return light->getStringName() == directional_light_string_name;
	})) != light_source_directions.end())
	{
		v3AverageLightDirection *= light_source_directions.size();
		light_source_directions.erase(light_to_remove_position);
		v3AverageLightDirection = (v3AverageLightDirection - (*light_to_remove_position)->getDirection()) / light_source_directions.size();
		v3CumulativeLightColor -= (*light_to_remove_position)->getColor();
		light_buffer.setClearColor(vec4{ v3CumulativeLightColor, 0.0f });
		return true;
	}

	return false;
}


void TransparentBox::removeAllLightSources()
{
	light_source_directions.clear();
	v3AverageLightDirection = vec3{ 0.0f };
	v3CumulativeLightColor = vec3{ 0.0f };
	light_buffer.setClearColor(vec4{ v3CumulativeLightColor, 0.0f });
}


void TransparentBox::setNumberOfPrimarySamples(uint32_t nsamples)
{
	num_primary_samples = nsamples;
}

void TransparentBox::setNumberOfSecondarySamples(uint32_t nsamples)
{
	num_secondary_samples = nsamples;
}


void TransparentBox::setSolidAngle(float fstangle)
{
	solid_angle = fstangle;
}


void TransparentBox::installPointCloud(const ImmutableTexture3D& _3d_point_cloud)
{
	if (!medium_texture_ref_code)
		medium_texture_ref_code = registerTexture(_3d_point_cloud, medium_sampler_ref_code);
	else
		updateTexture(medium_texture_ref_code, _3d_point_cloud, medium_sampler_ref_code);
}


void TransparentBox::installColormap(const ImmutableTexture1D& colormap)
{
	if (!colormap_texture_ref_code)
		colormap_texture_ref_code = registerTexture(colormap);
	else
		updateTexture(colormap_texture_ref_code, colormap);
}


void TransparentBox::useRGBChannel(bool enable_rgb_scattering_state)
{
	should_use_rgb_channel = enable_rgb_scattering_state ? 1 : 0;
}


void TransparentBox::useColormap(bool enable_colormap_state)
{
	should_use_colormap = enable_colormap_state ? 1 : 0;
}


void TransparentBox::setMediumUniformColor(const vec3& color)
{
	medium_color = color;
}


vec3 TransparentBox::getDimensions() const { return vec3{ width, height, depth }; }


uint32_t TransparentBox::getNumberOfSamples() const { return num_primary_samples; }


float TransparentBox::getSolidAngle() const { return solid_angle; }


bool TransparentBox::isRGBChannelInUse() const { return should_use_rgb_channel ? true : false; }


bool TransparentBox::isColormapInUse() const { return should_use_colormap ? true : false; }


vec3 TransparentBox::getMediumUniformColor() const { return medium_color; }


bool TransparentBox::supportsRenderingMode(uint32_t rendering_mode) const
{
	switch (rendering_mode)
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
		return true;
	default:
		return false;
	}
}


void TransparentBox::configureViewProjectionTransform(const AbstractProjectingDevice& projecting_device)
{
	//Save pointer to the projecting device used by the renderer
	p_renderer_projection = &projecting_device;
	v3ViewerLocation = projecting_device.getLocation();
	v3ViewerDirection = projecting_device.getViewDirection();
	v3ViewerUpVector = projecting_device.getUpVector();

	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
	{
		mat4 m4InvProj = projecting_device.getProjectionTransform().inverse();
		mat4 m4CameraToDimObj = (projecting_device.getViewTransform() * getObjectTransform()).inverse();

		vec4 v4Viewport;
		GLfloat viewport_params[4];
		glGetFloatv(GL_VIEWPORT, viewport_params);
		v4Viewport.x = viewport_params[0];
		v4Viewport.y = viewport_params[1];
		v4Viewport.z = viewport_params[2];
		v4Viewport.w = viewport_params[3];

		vec2 v2DepthRange;
		GLfloat depth_range_params[2];
		glGetFloatv(GL_DEPTH_RANGE, depth_range_params);
		v2DepthRange.x = depth_range_params[0];
		v2DepthRange.y = depth_range_params[1];

		mat4 m4MVP = projecting_device.getProjectionTransform() * projecting_device.getViewTransform() * 
			getObjectTransform() * getObjectScaleTransform();

		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformMatrix("m4MVP", m4MVP);
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v4Viewport", v4Viewport);
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v2DepthRange", v2DepthRange);
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3Scale", getObjectScale());
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformMatrix("m4InvProj", m4InvProj);
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformMatrix("m4CameraToDimObj", m4CameraToDimObj);

		break;
	}


	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
	{
		generate_proxy_geometry();

		//Determine light projection bounding box
		vec3 v3A1 = v3AverageLightDirection.get_normalized();

		std::array<vec3, 3> viewer_basis;
		viewer_basis[1] = projecting_device.getUpVector();
		viewer_basis[2] = -projecting_device.getViewDirection();
		viewer_basis[0] = viewer_basis[1].cross_product(viewer_basis[2]);

		vec3 v3A2 = *std::min_element(viewer_basis.begin(), viewer_basis.end(),
			[&v3A1](const vec3& elem1, const vec3& elem2) -> bool 
		{return std::abs(v3A1.dot_product(elem1)) < std::abs(v3A1.dot_product(elem2)); });
		float fAux = v3A1.dot_product(v3A2);
		if (fAux != 0.0f)
			v3A2 = (v3A1 - 1.0f / fAux*v3A2).get_normalized();

		vec3 v3A3 = v3A1.cross_product(v3A2);


		//Project view frustum onto the light projection bounding box
		float left, right, bottom, top, near, far;
		projecting_device.getProjectionVolume(&left, &right, &bottom, &top, &near, &far);

		//We cut the frustum so that its far plane is "far enough" to only contain the transparent box
		vec3 v3ObjectScale = getObjectScale();
		vec4 TransparentBoxFarPlane{ 0, 0, -depth / 2 * v3ObjectScale.z, 1.0f };
		TransparentBoxFarPlane = projecting_device.getViewTransform() * getObjectTransform() *TransparentBoxFarPlane;
		TransparentBoxFarPlane.x /= TransparentBoxFarPlane.w;
		TransparentBoxFarPlane.y /= TransparentBoxFarPlane.w;
		TransparentBoxFarPlane.z /= TransparentBoxFarPlane.w;
		far = -std::min(-2*near, std::max(-far, TransparentBoxFarPlane.z));

		std::array<vec3, 8> view_frustum = {
			vec3{ left, bottom, -near }, vec3{ right, bottom, -near }, vec3{ right, top, -near }, vec3{ left, top, -near }
		};
		view_frustum[4] = view_frustum[0] * far / near;
		view_frustum[5] = view_frustum[1] * far / near;
		view_frustum[6] = view_frustum[2] * far / near;
		view_frustum[7] = view_frustum[3] * far / near;

		vec3 v3ViewFrustumCenter = vec3{ (left + right) / 2.0f, (bottom + top) / 2.0f, -near } *(near + far) / (2.0f * near);	//compute center of the view frustum
		mat4 m4InvViewTransform = projecting_device.getViewTransform().inverse();
		vec4 v4Aux = m4InvViewTransform * vec4{ v3ViewFrustumCenter, 1.0f };
		v3ViewFrustumCenter = vec3{ v4Aux.x, v4Aux.y, v4Aux.z } / v4Aux.w;	//transform center of the view frustum into world-space coordinates

		//transform vertices of the view frustum into world-space coordinates and align the result with the origin of light frustum basis triplet
		std::transform(view_frustum.begin(), view_frustum.end(), view_frustum.begin(),
			[&m4InvViewTransform, &v3ViewFrustumCenter](const vec3& elem)->vec3
		{
			vec4 v4Aux = m4InvViewTransform * vec4{ elem, 1.0f };
			return vec3{ v4Aux.x, v4Aux.y, v4Aux.z } / v4Aux.w - v3ViewFrustumCenter;
		});

		//perform projection of the view frustum onto the light projection volume
		std::pair<std::array<vec3, 8>::const_iterator, std::array<vec3, 8>::const_iterator> A1_minmax =
			std::minmax_element(view_frustum.begin(), view_frustum.end(),
			[&v3A1](const vec3& elem1, const vec3& elem2) -> bool{ return v3A1.dot_product(elem1 - elem2) < 0; });
		float A1_min = v3A1.dot_product(*A1_minmax.first), A1_max = v3A1.dot_product(*A1_minmax.second);

		std::pair<std::array<vec3, 8>::const_iterator, std::array<vec3, 8>::const_iterator> A2_minmax =
			std::minmax_element(view_frustum.begin(), view_frustum.end(),
			[&v3A2](const vec3& elem1, const vec3& elem2) -> bool{ return v3A2.dot_product(elem1 - elem2) < 0; });
		float A2_min = v3A2.dot_product(*A2_minmax.first), A2_max = v3A2.dot_product(*A2_minmax.second);

		std::pair<std::array<vec3, 8>::const_iterator, std::array<vec3, 8>::const_iterator> A3_minmax =
			std::minmax_element(view_frustum.begin(), view_frustum.end(),
			[&v3A3](const vec3& elem1, const vec3& elem2) -> bool{ return v3A3.dot_product(elem1 - elem2) < 0; });
		float A3_min = v3A3.dot_product(*A3_minmax.first), A3_max = v3A3.dot_product(*A3_minmax.second);


		//Define light location and light projection bounding box
		vec3 v3LightLocation = v3ViewFrustumCenter + (A1_min - near) * v3A1;
		lightview_projection = OrthogonalProjectingDevice{ "transparent_box_lightview_projection",
			A3_min, A3_max, A2_min, A2_max, near, near + A1_max - A1_min, v3LightLocation, v3AverageLightDirection, v3A2 };

		//Apply projecting device to the rendering canvas
		canvas_filter.initialize();

		break;
	}

	}


	//Determine if the viewer is located inside of the transparent box
	float aux, near;
	projecting_device.getProjectionVolume(&aux, &aux, &aux, &aux, &near, &aux);
	vec4 v4ViewFocusInObjectSpace = (getObjectTransform()*getObjectScaleTransform()).inverse()*vec4{ v3ViewerLocation + near * v3ViewerDirection, 1.0f };
	v4ViewFocusInObjectSpace.x /= v4ViewFocusInObjectSpace.w;
	v4ViewFocusInObjectSpace.y /= v4ViewFocusInObjectSpace.w;
	v4ViewFocusInObjectSpace.z /= v4ViewFocusInObjectSpace.w;


	is_viewer_inside = 
		v4ViewFocusInObjectSpace.x >= -width / 2.0f && v4ViewFocusInObjectSpace.x <= width / 2.0f &&
		v4ViewFocusInObjectSpace.y >= -height / 2.0f && v4ViewFocusInObjectSpace.y <= height / 2.0f &&
		v4ViewFocusInObjectSpace.z >= -depth / 2.0f && v4ViewFocusInObjectSpace.z <= depth / 2.0f;
}


uint32_t TransparentBox::getNumberOfRenderingPasses(uint32_t rendering_mode) const
{
	switch(rendering_mode)
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
		return 1;

	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
		return 2 * num_primary_samples - 1 + 1;

	default:
		return 0;
	}
}


bool TransparentBox::configureRendering_RT(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	render_target.setPrimitiveRestartEnableState(true);
	render_target.setPrimitiveRestartIndexValue(0xFFFF);

	if (is_viewer_inside) render_target.setCullTestEnableState(false);
	else render_target.setCullTestEnableState(true);

	render_target.setColorBlendEnableState(true);
	render_target.setRGBSourceBlendFactor(ColorBlendFactor::SRC_ALPHA);
	render_target.setRGBDestinationBlendFactor(ColorBlendFactor::ONE_MINUS_SRC_ALPHA);
	render_target.setAlphaSourceBlendFactor(ColorBlendFactor::ZERO);
	render_target.setAlphaDestinationBlendFactor(ColorBlendFactor::ONE);
	render_target.setBlendEquation(ColorBlendEquation::ADD);

	render_target.setDepthBufferUpdateFlag(false);

	render_target.applyOpenGLContextSettings();


	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3BoxSize", vec3{ width, height, depth });
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("uiNumPrimarySamples", num_primary_samples);
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("uiNumSecondarySamples", num_secondary_samples);
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("fSolidAngle", solid_angle);
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("bUseRGBChannel", should_use_rgb_channel);
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("bUseColormap", should_use_colormap);
	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3MediumColor", medium_color);

	if (medium_texture_ref_code)
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("s3dMediumSampler", getBindingUnit(medium_texture_ref_code));

	if (colormap_texture_ref_code)
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("s1dColormapSampler", getBindingUnit(colormap_texture_ref_code));

	mat4 LightTransform = getObjectTransform().inverse();
	LightTransform[0][3] = 0;
	LightTransform[1][3] = 0;
	LightTransform[2][3] = 0;

	std::list<const DirectionalLight*>::const_iterator light_iterator = light_source_directions.begin();
	for (int i = 0; i < static_cast<int>(light_source_directions.size()); ++i)
	{
		vec4 aux = LightTransform * vec4{ (*light_iterator)->getDirection(), 1.0f };
		vec3 light_direction{ aux.x / aux.w, aux.y / aux.w, aux.z / aux.w };

		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3LightSourceDirections[" + std::to_string(i) + "]", light_direction);
		retrieveShaderProgram(rendering_program0_ref_code)->assignUniformVector("v3LightSourceIntensities[" + std::to_string(i) + "]", (*light_iterator)->getColor());

		light_iterator++;
	}

	retrieveShaderProgram(rendering_program0_ref_code)->assignUniformScalar("uiNumLightSources", static_cast<uint32_t>(light_source_directions.size()));

	glBindVertexArray(ogl_vertex_attribute_object0);

	if (medium_texture_ref_code)
		bindTexture(medium_texture_ref_code);

	if (colormap_texture_ref_code)
		bindTexture(colormap_texture_ref_code);

	if (getActiveRenderingMode() == TW_RENDERING_MODE_RAY_CAST_GAS)
		retrieveShaderProgram(rendering_program0_ref_code)->assignSubroutineUniform("func_optical_model", PipelineStage::FRAGMENT_SHADER, "GasScattering");

	if (getActiveRenderingMode() == TW_RENDERING_MODE_RAY_CAST_ABSORBENT)
		retrieveShaderProgram(rendering_program0_ref_code)->assignSubroutineUniform("func_optical_model", PipelineStage::FRAGMENT_SHADER, "EmissionAbsorption");

	COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program0_ref_code)).activate();

	return true;
}


bool TransparentBox::configureRendering_PG(uint32_t rendering_pass)
{
	Framebuffer* p_current_rendering_target = nullptr;

	//Depending on the pass number (even passes blend new data into the eye buffer, while odd passes update the light buffer) assign values to the corresponding uniform variables
	mat4 m4LightViewProjection = 
		lightview_projection.getProjectionTransform() * lightview_projection.getViewTransform() *
		getObjectTransform() * getObjectScaleTransform();
	if (rendering_pass % 2)
	{
		p_current_rendering_target = &light_buffer;
		retrieveShaderProgram(rendering_program1_ref_code)->assignUniformMatrix("m4LightViewProjection", m4LightViewProjection);
		retrieveShaderProgram(rendering_program1_ref_code)->assignUniformVector("v3MediumColor", medium_color);

		if (medium_texture_ref_code)
			retrieveShaderProgram(rendering_program1_ref_code)->assignUniformScalar("s3dMediumSampler", getBindingUnit(medium_texture_ref_code));

		if (colormap_texture_ref_code)
			retrieveShaderProgram(rendering_program1_ref_code)->assignUniformScalar("s1dColormapSampler", getBindingUnit(colormap_texture_ref_code));

		retrieveShaderProgram(rendering_program1_ref_code)->assignUniformScalar("bUseRGBChannel", should_use_rgb_channel);
		retrieveShaderProgram(rendering_program1_ref_code)->assignUniformScalar("bUseColormap", should_use_colormap);
		COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program1_ref_code)).activate();
	}
	else
	{
		p_current_rendering_target = &eye_buffer;
		if (is_viewer_inside)p_current_rendering_target->setCullTestEnableState(false);
		else p_current_rendering_target->setCullTestEnableState(true);

		mat4 m4MVP = p_renderer_projection->getProjectionTransform() * p_renderer_projection->getViewTransform() *
			getObjectTransform() * getObjectScaleTransform();
		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformMatrix("m4MVP", m4MVP);
		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformMatrix("m4LightViewProjection", m4LightViewProjection);
		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformScalar("s2dLightAttenuationBuffer", getBindingUnit(light_buffer_texture_ref_code));
		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformVector("v3MediumColor", medium_color);

		if (medium_texture_ref_code)
			retrieveShaderProgram(rendering_program2_ref_code)->assignUniformScalar("s3dMediumSampler", getBindingUnit(medium_texture_ref_code));

		if (colormap_texture_ref_code)
			retrieveShaderProgram(rendering_program2_ref_code)->assignUniformScalar("s1dColormapSampler", getBindingUnit(colormap_texture_ref_code));

		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformScalar("bUseRGBChannel", should_use_rgb_channel);
		retrieveShaderProgram(rendering_program2_ref_code)->assignUniformScalar("bUseColormap", should_use_colormap);
		COMPLETE_SHADER_PROGRAM_CAST(retrieveShaderProgram(rendering_program2_ref_code)).activate();

		if (light_buffer_texture_ref_code)
			bindTexture(light_buffer_texture_ref_code);
	}


	//inverse_proxy_geometry equals 'true' if proxy geometry slices shell be rendered "from back to front" using the Over operator
	bool inverse_proxy_geometry_orientation = v3ViewerDirection.dot_product(v3AverageLightDirection) < 0;

	if (inverse_proxy_geometry_orientation || rendering_pass % 2)
	{
		//Blending into the inverse eye buffer and into the light buffer is done using the Over operator
		p_current_rendering_target->setRGBSourceBlendFactor(ColorBlendFactor::ONE);
		p_current_rendering_target->setRGBDestinationBlendFactor(ColorBlendFactor::ONE_MINUS_SRC_ALPHA);
		p_current_rendering_target->setAlphaSourceBlendFactor(ColorBlendFactor::ONE);
		p_current_rendering_target->setAlphaDestinationBlendFactor(ColorBlendFactor::ONE_MINUS_SRC_ALPHA);
	}
	else
	{
		//Blending into the eye buffer is done using the Under operator
		p_current_rendering_target->setRGBSourceBlendFactor(ColorBlendFactor::ONE_MINUS_DST_ALPHA);
		p_current_rendering_target->setRGBDestinationBlendFactor(ColorBlendFactor::ONE);
		p_current_rendering_target->setAlphaSourceBlendFactor(ColorBlendFactor::ONE_MINUS_DST_ALPHA);
		p_current_rendering_target->setAlphaDestinationBlendFactor(ColorBlendFactor::ONE);
	}


	glBindVertexArray(ogl_vertex_attribute_object1);

	if (medium_texture_ref_code)
		bindTexture(medium_texture_ref_code);

	if (colormap_texture_ref_code)
		bindTexture(colormap_texture_ref_code);


	return true;
}


bool TransparentBox::configureRendering(AbstractRenderingDevice& render_target, uint32_t rendering_pass)
{
	//Save pointer to the target rendering device
	p_render_target = &render_target;
	render_target.pushOpenGLContextSettings();	//save current context settings of the render target

	if (rendering_pass >= getNumberOfRenderingPasses(getActiveRenderingMode())) return false;

	current_rendering_pass = rendering_pass;

	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
		//If target rendering device is not yet active, activate it
		if (!render_target.isActive())
			render_target.makeActive();
		return configureRendering_RT(render_target, rendering_pass);


	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
	{
		if (rendering_pass < 2 * num_primary_samples - 1)
		{
			if (rendering_pass % 2) light_buffer.makeActive();
			else eye_buffer.makeActive();
			bool rv = configureRendering_PG(rendering_pass);
			if (rendering_pass % 2) light_buffer.applyOpenGLContextSettings();
			else eye_buffer.applyOpenGLContextSettings();

			return rv;
		}
		else
		{
			render_target.makeActive();
			render_target.setColorBlendEnableState(true);
			render_target.setRGBSourceBlendFactor(ColorBlendFactor::SRC_ALPHA);
			render_target.setAlphaSourceBlendFactor(ColorBlendFactor::ONE);
			render_target.setRGBDestinationBlendFactor(ColorBlendFactor::ONE_MINUS_SRC_ALPHA);
			render_target.setAlphaDestinationBlendFactor(ColorBlendFactor::ONE);
			render_target.setRGBBlendEquation(ColorBlendEquation::ADD);
			render_target.setAlphaBlendEquation(ColorBlendEquation::MAX);
			render_target.applyOpenGLContextSettings();
		}

		return true;
	}

	default: return false;
	}

}


bool TransparentBox::render()
{
	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
		glDrawElements(GL_TRIANGLE_STRIP, 29, GL_UNSIGNED_SHORT, 0);
		return true;

	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
	{
		if (current_rendering_pass < 2*num_primary_samples - 1)
		{
			//Compute offset to the currently rendered slice
			uint32_t offset = 0;
			for (int i = 0; i < static_cast<int>(current_rendering_pass) / 2; ++i) offset += p_num_slice_vertices[i];

			//Check if current rendering pass is a light attenuation pass
			if (current_rendering_pass % 2 != 0)
			{
				if(current_rendering_pass == 1) light_buffer.clearBuffers(BufferClearTarget::COLOR_DEPTH);
				light_buffer.attachRenderer([offset, this](Framebuffer& framebuffer) -> void
				{
					glDrawArrays(GL_TRIANGLE_FAN, offset, p_num_slice_vertices[current_rendering_pass / 2]);
				});
				light_buffer.refresh();
			}
			else
			{
				if(current_rendering_pass == 0) eye_buffer.clearBuffers(BufferClearTarget::COLOR_DEPTH);
				eye_buffer.attachRenderer([offset, this](Framebuffer& framebuffer) -> void
				{
					glDrawArrays(GL_TRIANGLE_FAN, offset, p_num_slice_vertices[current_rendering_pass / 2]);
				});
				eye_buffer.refresh();
			}
		}
		else
		{
			canvas_filter.pass(*p_renderer_projection, *p_render_target);
		}

		return true;
	}

	default:
		return false;
	}
}


bool TransparentBox::configureRenderingFinalization()
{ 
	switch (getActiveRenderingMode())
	{
	case TW_RENDERING_MODE_RAY_CAST_GAS:
	case TW_RENDERING_MODE_RAY_CAST_ABSORBENT:
		p_render_target->popOpenGLContextSettings();
		break;

	case TW_RENDERING_MODE_PROXY_GEOMETRY_ABSORBENT:
		if (current_rendering_pass < 2 * num_primary_samples - 1)
		{
			p_render_target->popOpenGLContextSettings();
		}
		else
		{
			p_render_target->makeActive();
		}
	}

	return true;
}


