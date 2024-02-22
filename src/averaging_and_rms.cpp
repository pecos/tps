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
#include "averaging_and_rms.hpp"

#include <mfem/general/forall.hpp>

// TODO(kevin): add multi species and two temperature case.
Averaging::Averaging(RunConfiguration &_config) : Up(nullptr), mesh(nullptr), config(_config) {
  num_equation = 0;
  dim = 0;
  nvel = 0;
  numRMS = 0;

  sampleInterval = config.GetMeanSampleInterval();
  samplesMean = 0;
  samplesRMS = 0;
  startMean = config.GetMeanStartIter();
  computeMean = false;
  if (sampleInterval != 0) computeMean = true;

  // NB: Must call registerField to set up
  meanUp = nullptr;
}

Averaging::~Averaging() {
  if (computeMean) {
    delete paraviewMean;
    delete meanP;
    delete meanV;
    delete meanRho;
    delete rms;
    delete meanUp;
    delete rmsFes;
  }
}

void Averaging::registerField(ParGridFunction *field_to_average) {
  Up = field_to_average;
  mesh = Up->ParFESpace()->GetParMesh();

  num_equation = Up->ParFESpace()->GetVDim();
  dim = mesh->Dimension();
  nvel = (config.isAxisymmetric() ? 3 : mesh->Dimension());

  // Always assume 6 components of the Reynolds stress tensor
  numRMS = 6;

  if (computeMean) {
    meanUp = new ParGridFunction(Up->ParFESpace());

    rmsFes = new ParFiniteElementSpace(mesh, Up->ParFESpace()->FEColl(), numRMS, Ordering::byNODES);
    rms = new ParGridFunction(rmsFes);

    initiMeanAndRMS();
  }
}

void Averaging::initializeViz(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes) {
  if (computeMean) {
    assert(meanUp != nullptr);

    // "helper" spaces to index into meanUp
    meanRho = new ParGridFunction(fes, meanUp->GetData());
    meanV = new ParGridFunction(dfes, meanUp->GetData() + fes->GetNDofs());
    meanP = new ParGridFunction(fes, meanUp->GetData() + (1 + nvel) * fes->GetNDofs());

    // ParaviewMean
    string name = "mean_";
    name.append(config.GetOutputName());
    paraviewMean = new ParaViewDataCollection(name, mesh);
    paraviewMean->SetLevelsOfDetail(config.GetSolutionOrder());
    paraviewMean->SetHighOrderOutput(true);
    paraviewMean->SetPrecision(8);

    paraviewMean->RegisterField("dens", meanRho);
    paraviewMean->RegisterField("vel", meanV);
    paraviewMean->RegisterField("press", meanP);
    paraviewMean->RegisterField("rms", rms);
  }
}

void Averaging::addSampleMean(const int &iter, GasMixture *mixture) {
  if (computeMean) {
    assert(meanUp != nullptr);
    if (iter % sampleInterval == 0 && iter >= startMean) {
      if (iter == startMean && (mesh->GetMyRank() == 0)) cout << "Starting mean calculation." << endl;

      if (samplesRMS == 0) {
        *rms = 0.0;
      }

      addSample(mixture);
      samplesMean++;
      samplesRMS++;
    }
  }
}

void Averaging::write_meanANDrms_restart_files(const int &iter, const double &time) {
  if (computeMean) {
    if (config.isMeanHistEnabled()) {
      paraviewMean->SetCycle(iter);
      paraviewMean->SetTime(time);
    }

    paraviewMean->Save();
  }
}

void Averaging::initiMeanAndRMS() {
  assert(meanUp != nullptr);
  assert(rms != nullptr);
  *meanUp = 0.0;
  *rms = 0.0;
}

void Averaging::addSample(GasMixture *mixture) {
  double *d_meanUp = meanUp->ReadWrite();
  double *d_rms = rms->ReadWrite();
  const double *d_Up = Up->Read();

  double dSamplesMean = (double)samplesMean;
  double dSamplesRMS = (double)samplesRMS;

  GasMixture *d_mixture = mixture;

  const int Ndof = meanUp->ParFESpace()->GetNDofs();
  const int d_dim = dim;
  const int d_neqn = num_equation;

  MFEM_FORALL(n, Ndof, {
    double meanVel[gpudata::MAXDIM], vel[gpudata::MAXDIM];  // double meanVel[3], vel[3];
    // double nUp[20];  // NOTE: lets make sure we don't have more than 20 eq.
    // NOTE(kevin): (presumably) marc left this hidden note here..
    double nUp[gpudata::MAXEQUATIONS];

    for (int eq = 0; eq < d_neqn; eq++) {
      nUp[eq] = d_Up[n + eq * Ndof];
    }

    // mean
    for (int eq = 0; eq < d_neqn; eq++) {
      double mUpi = d_meanUp[n + eq * Ndof];
      double mVal = dSamplesMean * mUpi;

      double newMeanUp;
      if (eq != 1 + d_dim) {
        newMeanUp = (mVal + nUp[eq]) / (dSamplesMean + 1);
      } else {  // eq == 1+d_dim
        double p = nUp[eq];
        if (mixture != nullptr) {
          p = d_mixture->ComputePressureFromPrimitives(nUp);
        }
        newMeanUp = (mVal + p) / (dSamplesMean + 1);
      }

      d_meanUp[n + eq * Ndof] = newMeanUp;
    }

    // fill out mean velocity array
    for (int d = 0; d < d_dim; d++) {
      meanVel[d] = d_meanUp[n + (d + 1) * Ndof];
      vel[d] = nUp[d + 1];
    }
    if (d_dim != 3) {
      meanVel[2] = 0.;
      vel[2] = 0.;
    }

    // ----- RMS -----
    double val = 0.;
    // xx
    val = d_rms[n];
    d_rms[n] = (val * dSamplesRMS + (vel[0] - meanVel[0]) * (vel[0] - meanVel[0])) / (dSamplesRMS + 1);

    // yy
    val = d_rms[n + Ndof];
    d_rms[n + Ndof] = (val * dSamplesRMS + (vel[1] - meanVel[1]) * (vel[1] - meanVel[1])) / (dSamplesRMS + 1);

    // zz
    val = d_rms[n + 2 * Ndof];
    d_rms[n + 2 * Ndof] = (val * dSamplesRMS + (vel[2] - meanVel[2]) * (vel[2] - meanVel[2])) / (dSamplesRMS + 1);

    // xy
    val = d_rms[n + 3 * Ndof];
    d_rms[n + 3 * Ndof] = (val * dSamplesRMS + (vel[0] - meanVel[0]) * (vel[1] - meanVel[1])) / (dSamplesRMS + 1);

    // xz
    val = d_rms[n + 4 * Ndof];
    d_rms[n + 4 * Ndof] = (val * dSamplesRMS + (vel[0] - meanVel[0]) * (vel[2] - meanVel[2])) / (dSamplesRMS + 1);

    // yz
    val = d_rms[n + 5 * Ndof];
    d_rms[n + 5 * Ndof] = (val * dSamplesRMS + (vel[1] - meanVel[1]) * (vel[2] - meanVel[2])) / (dSamplesRMS + 1);
  });
}
