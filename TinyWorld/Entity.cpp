#include "Entity.h"

using namespace tiny_world;


unsigned long long Entity::internal_id_counter = 0;


Entity::Entity(const std::string& class_string_name) : internal_id{ ++internal_id_counter }, 
class_string_name(class_string_name), string_name(class_string_name + "#" + std::to_string(internal_id))
{

}


Entity::Entity(const std::string& class_string_name, const std::string& entity_string_name) : internal_id{ ++internal_id_counter },
class_string_name(class_string_name), string_name(entity_string_name)
{

}


Entity::Entity(const Entity& other) : ErrorBehavioral{ other }, internal_id{ ++internal_id_counter },
class_string_name(other.class_string_name), string_name(other.string_name)
{

}


Entity::Entity(Entity&& other) : ErrorBehavioral{ std::move(other) }, internal_id{ other.internal_id },
class_string_name(std::move(other.class_string_name)), string_name(std::move(other.string_name))
{
	
}


Entity::~Entity()
{

}


Entity& Entity::operator=(const Entity& other)
{
	if (this == &other)
		return *this;

	ErrorBehavioral::operator=(other);

	string_name = other.string_name;

	return *this;
}


Entity& Entity::operator=(Entity&& other)
{
	if (this == &other)
		return *this;

	ErrorBehavioral::operator=(std::move(other));

	string_name = std::move(other.string_name);

	return *this;
}


unsigned long long Entity::getId() const { return internal_id; }


std::string Entity::getStringName() const { return string_name; }


void Entity::setStringName(const std::string& entity_string_name) { string_name = entity_string_name; }


std::string Entity::getClassStringName() const { return class_string_name; }