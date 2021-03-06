// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2022, The PECOS Development Team, University of Texas at Austin
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------------el-
#ifndef IO_HPP_
#define IO_HPP_

#include <vector>

#include "tps_mfem_wrap.hpp"

class IOFamily {
 public:
  std::string description_;       // family description
  std::string group_;             // HDF5 group name
  mfem::ParGridFunction *pfunc_;  // pargrid function owning the data for this IO family

  bool allowsAuxRestart;

  // true if the family is to be found in restart files
  bool inReastartFile;

  mfem::FiniteElementSpace *serial_fes;
  mfem::GridFunction *serial_sol;
};

class IOVar {
 public:
  std::string varName_;  // solution variable
  int index_;            // variable index in the pargrid function
};

class IODataOrganizer {
 public:
  std::vector<IOFamily> families_;                  // registered IO families
  std::map<std::string, std::vector<IOVar>> vars_;  // solution var info for each IO family

  void registerIOFamily(std::string description, std::string group, mfem::ParGridFunction *pfunc,
                        bool auxRestart = true, bool inRestartFile = true);
  ~IODataOrganizer();

  void registerIOVar(std::string group, std::string varName, int index);
  int getIOFamilyIndex(std::string group);

  void initializeSerial(bool root, bool serial, mfem::Mesh *serial_mesh);
};

#endif  // IO_HPP_
