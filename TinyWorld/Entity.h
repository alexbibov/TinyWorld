#ifndef TW__ENTITY__

#include <string>
#include "ErrorBehavioral.h"

namespace tiny_world
{
	class Entity : public ErrorBehavioral
	{
	private:
		static unsigned long long internal_id_counter;	//internal counter of the entity identifiers

		unsigned long long internal_id;	//internal identifier of the entity
		std::string class_string_name;	//string name of the class of objects to which the entity belongs
		std::string string_name;	//string name of the entity

	protected:
		Entity(const std::string& class_string_name);	//default initialization of an entity
		Entity(const Entity& other);	//copy initialization
		Entity(Entity&& other);	//move initialization
		Entity(const std::string& class_string_name, const std::string& entity_string_name);		//initializes new entity using provided string name

		Entity& operator=(const Entity& other);	//copy assignment
		Entity& operator=(Entity&& other); //move assignment
	public:

		unsigned long long getId() const;	//returns internal identifier of the entity

		std::string getStringName() const;	//returns string name of the entity, where the string name acts as "the weak identifier" of the entity
		void setStringName(const std::string& entity_string_name);	//sets new string name for the entity

		std::string getClassStringName() const;	//returns string name of the class to which the entity belongs


		~Entity();	//destructor
	};
}

#define TW__ENTITY__
#endif