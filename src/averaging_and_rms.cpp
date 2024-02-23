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

Averaging::Averaging(RunConfiguration &_config) : config(_config) {
  sampleInterval = config.GetMeanSampleInterval();
  samplesMean = 0;
  samplesRMS = 0;
  startMean = config.GetMeanStartIter();
  computeMean = false;
  if (sampleInterval != 0) computeMean = true;
  rank0_ = false;
}

Averaging::~Averaging() {
  if (computeMean) {
    delete paraviewMean;
    delete meanP;
    delete meanV;
    delete meanRho;
  }
}

void Averaging::registerField(ParGridFunction *field_to_average, bool compute_rms, int rms_start_index,
                              int rms_components) {
  // quick return if not computing stats...
  if (!computeMean) return;

  // otherwise, set up ParGridFunction to hold mean...
  ParMesh *mesh = field_to_average->ParFESpace()->GetParMesh();
  rank0_ = (mesh->GetMyRank() == 0);

  ParGridFunction *meanUp = new ParGridFunction(field_to_average->ParFESpace());
  *meanUp = 0.0;

  // and maybe the rms
  ParGridFunction *rms = nullptr;
  if (compute_rms) {
    // make sure incoming field has enough components to satisfy rms request
    assert((rms_start_index + rms_components) <= field_to_average->ParFESpace()->GetVDim());

    const int numRMS = rms_components * (rms_components + 1) / 2;

    const FiniteElementCollection *fec = field_to_average->ParFESpace()->FEColl();
    const int order = fec->GetOrder();

    FiniteElementCollection *rms_fec = fec->Clone(order);
    ParFiniteElementSpace *rmsFes = new ParFiniteElementSpace(mesh, rms_fec, numRMS, Ordering::byNODES);
    rms = new ParGridFunction(rmsFes);
    rms->MakeOwner(rms_fec);

    *rms = 0.0;
  }

  // and store those fields in an AveragingFamily object that gets appended to the avg_families_ vector
  avg_families_.emplace_back(AveragingFamily(field_to_average, meanUp, rms, rms_start_index, rms_components));
}

void Averaging::initializeViz(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes) {
  if (computeMean) {
    assert(avg_families_.size() == 1);

    ParGridFunction *meanUp = avg_families_[0].mean_fcn_;
    ParGridFunction *rms = avg_families_[0].rms_fcn_;

    ParMesh *mesh = meanUp->ParFESpace()->GetParMesh();
    const int nvel = (config.isAxisymmetric() ? 3 : mesh->Dimension());

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
    assert(avg_families_.size() >= 1);
    if (iter % sampleInterval == 0 && iter >= startMean) {
      if (iter == startMean && rank0_) cout << "Starting mean calculation." << endl;

      if (samplesRMS == 0) {
        // *rms = 0.0;
        *avg_families_[0].rms_fcn_ = 0.0;
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

void Averaging::addSample(GasMixture *mixture) {
  // Assert that there is something to average.  In principle we don't
  // need this, b/c the loop below is a no-op if there are no
  // families.  However, if you got to this point, you're expecting to
  // average something, so if there is nothing to average, we should
  // complain rather than silently do nothing.
  assert(avg_families_.size() >= 1);

  // Loop through families that have been registered and compute means and variances
  for (size_t ifam = 0; ifam < avg_families_.size(); ifam++) {
    double dSamplesMean = (double)samplesMean;
    double dSamplesRMS = (double)samplesRMS;

    GasMixture *d_mixture = mixture;

    AveragingFamily &fam = avg_families_[ifam];

    ParGridFunction *meanUp = fam.mean_fcn_;
    ParGridFunction *rms = fam.rms_fcn_;

    const double *d_Up = fam.instantaneous_fcn_->Read();

    double *d_meanUp = meanUp->ReadWrite();
    double *d_rms = rms->ReadWrite();

    ParMesh *mesh = meanUp->ParFESpace()->GetParMesh();

    const int d_neqn = meanUp->ParFESpace()->GetVDim();
    const int d_dim = mesh->Dimension();

    const int d_rms_start = fam.rms_start_index_;
    const int d_rms_components = fam.rms_components_;

    const int Ndof = meanUp->ParFESpace()->GetNDofs();

    MFEM_FORALL(n, Ndof, {
      // double meanVel[gpudata::MAXDIM], vel[gpudata::MAXDIM];  // double meanVel[3], vel[3];
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

      if (d_rms != nullptr) {
        double mean_state[gpudata::MAXEQUATIONS];

        for (int eq = 0; eq < d_neqn; eq++) {
          mean_state[eq] = d_meanUp[n + eq * Ndof];
        }

        // // fill out mean velocity array
        // for (int d = 0; d < d_dim; d++) {
        //   meanVel[d] = d_meanUp[n + (d + 1) * Ndof];
        //   vel[d] = nUp[d + 1];
        // }
        // if (d_dim != 3) {
        //   meanVel[2] = 0.;
        //   vel[2] = 0.;
        // }

        // ----- RMS -----
        double val = 0.;
        double delta_i = 0.;
        double delta_j = 0.;
        int rms_index = 0;

        // Variances first
        for (int i = d_rms_start; i < d_rms_start + d_rms_components; i++) {
          val = d_rms[n + rms_index * Ndof];
          delta_i = (nUp[i] - mean_state[i]);
          d_rms[n + rms_index * Ndof] = (val * dSamplesRMS + delta_i * delta_i) / (dSamplesRMS + 1);
          rms_index++;
        }

        // // NB: Leaving this here temporarily b/c even in 2-D, TPS
        // // keeps the "zz" component of the velocity rms

        // // TODO(trevilo): Eliminate this and update reference fields
        // // in regression tests once we have checked all the non-zero
        // // fields when computed with the refactored approach zz
        // val = d_rms[n + 2 * Ndof];
        // d_rms[n + 2 * Ndof] = (val * dSamplesRMS + (vel[2] - meanVel[2]) * (vel[2] - meanVel[2])) / (dSamplesRMS +
        // 1); rms_index++;

        // Covariances second
        for (int i = d_rms_start; i < d_rms_start + d_rms_components - 1; i++) {
          for (int j = d_rms_start + 1; j < d_rms_start + d_rms_components; j++) {
            val = d_rms[n + rms_index * Ndof];
            delta_i = (nUp[i] - mean_state[i]);
            delta_j = (nUp[j] - mean_state[j]);
            d_rms[n + rms_index * Ndof] = (val * dSamplesRMS + delta_i * delta_j) / (dSamplesRMS + 1);
            rms_index++;
          }
        }

        // // xy
        // val = d_rms[n + 3 * Ndof];
        // d_rms[n + 3 * Ndof] = (val * dSamplesRMS + (vel[0] - meanVel[0]) * (vel[1] - meanVel[1])) / (dSamplesRMS +
        // 1);

        // // xz
        // val = d_rms[n + 4 * Ndof];
        // d_rms[n + 4 * Ndof] = (val * dSamplesRMS + (vel[0] - meanVel[0]) * (vel[2] - meanVel[2])) / (dSamplesRMS +
        // 1);

        // // yz
        // val = d_rms[n + 5 * Ndof];
        // d_rms[n + 5 * Ndof] = (val * dSamplesRMS + (vel[1] - meanVel[1]) * (vel[2] - meanVel[2])) / (dSamplesRMS +
        // 1);
      }
    });
  }
}
