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

void Averaging::registerField(const ParGridFunction *field_to_average, bool compute_rms, int rms_start_index,
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
    // Extract fields for use on device (when available)
    AveragingFamily &fam = avg_families_[ifam];

    const ParGridFunction *inst = fam.instantaneous_fcn_;
    ParGridFunction *mean = fam.mean_fcn_;
    ParGridFunction *vari = fam.rms_fcn_;

    const double *d_inst = inst->Read();
    double *d_mean = mean->ReadWrite();
    double *d_vari = vari->ReadWrite();

    // Get mixture class pointer for use on device
    //
    // This is here to support original M2ulPhyS usage, where the
    // instantaneous field contains temperature, but the averaged
    // quantity is pressure.  But, in general, it may be null, in
    // which case, whatever data are in the state are averaged
    // directly.
    GasMixture *d_mixture = mixture;

    // Extract size information for use on device
    const int dof = mean->ParFESpace()->GetNDofs();                 // dofs per scalar field
    const int neq = mean->ParFESpace()->GetVDim();                  // number of scalar variables in mean field
    const int dim = mean->ParFESpace()->GetParMesh()->Dimension();  // spatial dimension

    const int d_rms_start = fam.rms_start_index_;
    const int d_rms_components = fam.rms_components_;

    // Extract sample size information for use on device
    double dSamplesMean = (double)samplesMean;
    double dSamplesRMS = (double)samplesRMS;

    // "Loop" over all dofs and update statistics
    MFEM_FORALL(n, dof, {
      // NB: This is only necessary when mixture is not nulltpr
      double nUp[gpudata::MAXEQUATIONS];
      for (int eq = 0; eq < neq; eq++) {
        nUp[eq] = d_inst[n + eq * dof];
      }

      // Update mean
      for (int eq = 0; eq < neq; eq++) {
        const double uinst = d_inst[n + eq * dof];
        const double umean = d_mean[n + eq * dof];
        const double N_umean = dSamplesMean * umean;

        if (eq != 1 + dim) {
          d_mean[n + eq * dof] = (N_umean + uinst) / (dSamplesMean + 1);
        } else {  // eq == 1+dim
          double p = nUp[eq];
          if (mixture != nullptr) {
            p = d_mixture->ComputePressureFromPrimitives(nUp);
          }
          d_mean[n + eq * dof] = (N_umean + p) / (dSamplesMean + 1);
        }
      }

      // Update variances (only computed if we have a place to put them)
      if (d_vari != nullptr) {
        double mean_state[gpudata::MAXEQUATIONS];

        for (int eq = 0; eq < neq; eq++) {
          mean_state[eq] = d_mean[n + eq * dof];
        }

        double val = 0.;
        double delta_i = 0.;
        double delta_j = 0.;
        int rms_index = 0;

        // Variances first (i.e., diagonal of the covariance matrix)
        for (int i = d_rms_start; i < d_rms_start + d_rms_components; i++) {
          val = d_vari[n + rms_index * dof];
          delta_i = (nUp[i] - mean_state[i]);
          d_vari[n + rms_index * dof] = (val * dSamplesRMS + delta_i * delta_i) / (dSamplesRMS + 1);
          rms_index++;
        }

        // Covariances second (i.e., off-diagonal components, if any)
        for (int i = d_rms_start; i < d_rms_start + d_rms_components - 1; i++) {
          for (int j = d_rms_start + 1; j < d_rms_start + d_rms_components; j++) {
            val = d_vari[n + rms_index * dof];
            delta_i = (nUp[i] - mean_state[i]);
            delta_j = (nUp[j] - mean_state[j]);
            d_vari[n + rms_index * dof] = (val * dSamplesRMS + delta_i * delta_j) / (dSamplesRMS + 1);
            rms_index++;
          }
        }
      }  // end variance
    });
  }
}
