#ifndef IO
#define IO

class IOFamily
{
public:
  std::string description_;     // family description
  std::string group_;           // HDF5 group name
  ParGridFunction *pfunc_;      // pargrid function owning the data for this IO family
};

class IOVar
{
public:
  std::string varName_;         // solution variable
  int index_;                   // variable index in the pargrid function
};

class IODataOrganizer
{
public:
  std::vector<IOFamily> families_;      // registered IO families
  std::map<string,vector<IOVar>> vars_; // solution var info for each IO family

  void registerIOFamily(std::string description, std::string group, ParGridFunction *pfunc);
  void registerIOVar   (std::string group, std::string varName, int index);
  int  getIOFamilyIndex(std::string group);
};

#endif
