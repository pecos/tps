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
#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include "tps_mfem_wrap.hpp"
#include "utils.hpp"

using namespace std;

namespace TPS {

// Forward declare Tps2Bolzmann class
class Tps2Boltzmann;

// Base class shared by all solver implementations
class Solver {
 private:
  std::string name_;
  std::string description_;

 public:
  // methods to be (optionally) implemented by derived solver classes
  virtual int getStatus() { return NORMAL; }
  virtual void initialize() { return; }
  virtual ~Solver() {}
  // methods *required* to be implemented by derived solver classes
  virtual void parseSolverOptions() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }
  virtual void solve() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }
  virtual void solveStep() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }
  virtual void solveBegin() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }
  virtual void solveEnd() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }
  virtual void visualization() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }

  /// Initialize the interface
  virtual void initInterface(Tps2Boltzmann &interface) {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }

  /// Push solver variables to interface
  virtual void push(Tps2Boltzmann &interface) {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }

  /// Fetch solver variables from interface
  virtual void fetch(Tps2Boltzmann &interface) {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }

  /// Get the mesh used by this Solver
  virtual mfem::ParMesh *getMesh() const {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
    return nullptr;
  }

  virtual const mfem::FiniteElementCollection *getFEC() const {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
    return nullptr;
  }

  virtual mfem::ParFiniteElementSpace *getFESpace() const {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
    return nullptr;
  }
};

// Base class for flow solver implementations
class PlasmaSolver : public Solver {
 public:
  virtual ~PlasmaSolver() {}

  /// Fetch the plasma electrical conductivity grid function
  virtual mfem::ParGridFunction *getPlasmaConductivityGF() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
    return nullptr;
  }

  /// Update the plasma conductivity
  virtual void evaluatePlasmaConductivityGF() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
  }

  /// Fetch the Joule heating grid function
  virtual mfem::ParGridFunction *getJouleHeatingGF() {
    cout << "ERROR: " << __func__ << " remains unimplemented" << endl;
    exit(1);
    return nullptr;
  }
};

}  // end namespace TPS

#endif  // SOLVER_HPP_
