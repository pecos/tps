// -----------------------------------------------------------------------------------bl-
// BSD 3-Clause License
//
// Copyright (c) 2020-2021, The PECOS Development Team, University of Texas at Austin
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
#ifndef FORCING_TERMS
#define FORCING_TERMS

#include <mfem.hpp>
#include <tps_config.h>
#include "run_configuration.hpp"

#ifdef _MASA_
#include "masa_handler.hpp"
#endif

using namespace mfem;
using namespace std;

class ForcingTerms
{
protected:
  double time;
  
  const int &dim;
  const int &num_equation;
  const int &order;
  const int &intRuleType;
  IntegrationRules *intRules;
  ParFiniteElementSpace *vfes;
  ParGridFunction *Up;
  ParGridFunction *gradUp;
  
  
  // added term
  ParGridFunction *b;
  
public:
  ForcingTerms( const int &_dim,
                const int &_num_equation,
                const int &_order,
                const int &_intRuleType,
                IntegrationRules *_intRules,
                ParFiniteElementSpace *_vfes,
                ParGridFunction *_Up,
                ParGridFunction *_gradUp );
  ~ForcingTerms();

  void setTime(double _time) { time = _time; }
  virtual void updateTerms() = 0;
  virtual void addForcingIntegrals(Vector &in);
};

// Constant pressure gradient term 
class ConstantPressureGradient: public ForcingTerms
{
private:
  //RunConfiguration &config;
  double pressGrad[3];
  
public:
  ConstantPressureGradient( const int &_dim,
                            const int &_num_equation,
                            const int &_order,
                            const int &_intRuleType,
                            IntegrationRules *_intRules,
                            ParFiniteElementSpace *_vfes,
                            ParGridFunction *_Up,
                            ParGridFunction *_gradUp,
                            RunConfiguration &_config );
  
  // Terms do not need updating
  virtual void updateTerms();
  
  //virtual void addForcingIntegrals(Vector &in);
};

#ifdef _MASA_
// Manufactured Solution using MASA
class MASA_forcings: public ForcingTerms
{
private:

public:
  MASA_forcings(const int &_dim,
                const int &_num_equation,
                const int &_order,
                const int &_intRuleType,
                IntegrationRules *_intRules,
                ParFiniteElementSpace *_vfes,
                ParGridFunction *_Up,
                ParGridFunction *_gradUp,
                RunConfiguration &_config );

  virtual void updateTerms();
};
#endif // _MASA_

#endif // FORCING_TERMS
