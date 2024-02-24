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

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"

Averaging::Averaging(RunConfiguration &config) {
  rank0_ = false;
  compute_mean_ = false;

  sample_interval_ = config.GetMeanSampleInterval();
  step_start_mean_ = config.GetMeanStartIter();

  if (sample_interval_ != 0) compute_mean_ = true;

  ns_mean_ = 0;
  ns_vari_ = 0;

  mean_output_name_ = "mean_";
  mean_output_name_.append(config.GetOutputName());
}

Averaging::~Averaging() {
  if (compute_mean_) {
    delete pvdc_;
    delete meanP;
    delete meanV;
    delete meanRho;
  }
}

void Averaging::registerField(std::string name, const ParGridFunction *field_to_average, bool compute_rms,
                              int rms_start_index, int rms_components) {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  // otherwise, set up ParGridFunction to hold mean...
  ParMesh *mesh = field_to_average->ParFESpace()->GetParMesh();
  rank0_ = (mesh->GetMyRank() == 0);

  ParGridFunction *mean = new ParGridFunction(field_to_average->ParFESpace());
  *mean = 0.0;

  // and maybe the rms
  ParGridFunction *vari = nullptr;
  if (compute_rms) {
    // make sure incoming field has enough components to satisfy rms request
    assert((rms_start_index + rms_components) <= field_to_average->ParFESpace()->GetVDim());

    const int num_variance = rms_components * (rms_components + 1) / 2;

    const FiniteElementCollection *fec = field_to_average->ParFESpace()->FEColl();
    const int order = fec->GetOrder();

    FiniteElementCollection *vari_fec = fec->Clone(order);
    ParFiniteElementSpace *vari_fes = new ParFiniteElementSpace(mesh, vari_fec, num_variance, Ordering::byNODES);
    vari = new ParGridFunction(vari_fes);
    vari->MakeOwner(vari_fec);

    *vari = 0.0;
  }

  // and store those fields in an AveragingFamily object that gets appended to the avg_families_ vector
  avg_families_.emplace_back(AveragingFamily(name, field_to_average, mean, vari, rms_start_index, rms_components));
}

void Averaging::initializeViz() {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  assert(avg_families_.size() >= 1);

  // Loop through the families and add them to the paraview output
  for (size_t i = 0; i < avg_families_.size(); i++) {
    ParGridFunction *mean = avg_families_[i].mean_fcn_;
    ParGridFunction *vari = avg_families_[i].rms_fcn_;

    const FiniteElementCollection *fec = mean->ParFESpace()->FEColl();
    const int order = fec->GetOrder();

    ParMesh *mesh = mean->ParFESpace()->GetParMesh();

    // If not yet allocated paraview, do it
    if (pvdc_ == nullptr) {
      // TODO(trevilo): This approach ASSUMES that all stats are on
      // the same mesh.  This is true for now, but should be
      // generalized
      pvdc_ = new ParaViewDataCollection(mean_output_name_, mesh);
      pvdc_->SetLevelsOfDetail(order);
      pvdc_->SetHighOrderOutput(true);
      pvdc_->SetPrecision(8);
    }

    // Register fields with paraview

    // TODO(trevilo): Add a name to AveragingFamily so we can use it here
    std::string mean_name("mean_");
    mean_name += avg_families_[i].name_;
    pvdc_->RegisterField(mean_name.c_str(), mean);

    if (vari != nullptr) {
      std::string vari_name("vari_");
      vari_name += avg_families_[i].name_;
      pvdc_->RegisterField(vari_name.c_str(), vari);
    }
  }
}

void Averaging::initializeVizForM2ulPhyS(ParFiniteElementSpace *fes, ParFiniteElementSpace *dfes, int nvel) {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  assert(avg_families_.size() == 1);

  ParGridFunction *meanUp = avg_families_[0].mean_fcn_;
  ParGridFunction *rms = avg_families_[0].rms_fcn_;

  const FiniteElementCollection *fec = meanUp->ParFESpace()->FEColl();
  const int order = fec->GetOrder();

  ParMesh *mesh = meanUp->ParFESpace()->GetParMesh();

  // "helper" spaces to index into meanUp
  meanRho = new ParGridFunction(fes, meanUp->GetData());
  meanV = new ParGridFunction(dfes, meanUp->GetData() + fes->GetNDofs());
  meanP = new ParGridFunction(fes, meanUp->GetData() + (1 + nvel) * fes->GetNDofs());

  // Register fields with paraview
  pvdc_ = new ParaViewDataCollection(mean_output_name_, mesh);
  pvdc_->SetLevelsOfDetail(order);
  pvdc_->SetHighOrderOutput(true);
  pvdc_->SetPrecision(8);

  pvdc_->RegisterField("dens", meanRho);
  pvdc_->RegisterField("vel", meanV);
  pvdc_->RegisterField("press", meanP);
  pvdc_->RegisterField("rms", rms);
}

void Averaging::addSample(const int &iter, GasMixture *mixture) {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  assert(avg_families_.size() >= 1);

  if (iter % sample_interval_ == 0 && iter >= step_start_mean_) {
    if (iter == step_start_mean_ && rank0_) cout << "Starting mean calculation." << endl;

    if (ns_vari_ == 0) {
      // If we got here, then either this is our first time averaging
      // or the variances have been restarted.  Either way, valid to
      // set variances to zero.
      for (size_t ifam = 0; ifam < avg_families_.size(); ifam++) {
        *avg_families_[ifam].rms_fcn_ = 0.0;
      }
    }

    if (mixture != nullptr) {
      addSampleInternal(mixture);
    } else {
      addSampleInternal();
    }
    ns_mean_++;
    ns_vari_++;
  }
}

void Averaging::writeViz(const int &iter, const double &time, bool save_mean_hist) {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  if (save_mean_hist) {
    pvdc_->SetCycle(iter);
    pvdc_->SetTime(time);
  }

  pvdc_->Save();
}

void Averaging::addSampleInternal() {
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

    // Extract size information for use on device
    const int dof = mean->ParFESpace()->GetNDofs();  // dofs per scalar field
    const int neq = mean->ParFESpace()->GetVDim();   // number of scalar variables in mean field

    const int d_rms_start = fam.rms_start_index_;
    const int d_rms_components = fam.rms_components_;

    // Extract sample size information for use on device
    double dSamplesMean = (double)ns_mean_;
    double dSamplesRMS = (double)ns_vari_;

    // "Loop" over all dofs and update statistics
    MFEM_FORALL(n, dof, {
      // Update mean
      for (int eq = 0; eq < neq; eq++) {
        const double uinst = d_inst[n + eq * dof];
        const double umean = d_mean[n + eq * dof];
        const double N_umean = dSamplesMean * umean;
        d_mean[n + eq * dof] = (N_umean + uinst) / (dSamplesMean + 1);
      }

      // Update variances (only computed if we have a place to put them)
      if (d_vari != nullptr) {
        double val = 0.;
        double delta_i = 0.;
        double delta_j = 0.;
        int rms_index = 0;

        // Variances first (i.e., diagonal of the covariance matrix)
        for (int i = d_rms_start; i < d_rms_start + d_rms_components; i++) {
          const double uinst = d_inst[n + i * dof];
          const double umean = d_mean[n + i * dof];
          delta_i = (uinst - umean);
          val = d_vari[n + rms_index * dof];
          d_vari[n + rms_index * dof] = (val * dSamplesRMS + delta_i * delta_i) / (dSamplesRMS + 1);
          rms_index++;
        }

        // Covariances second (i.e., off-diagonal components, if any)
        for (int i = d_rms_start; i < d_rms_start + d_rms_components - 1; i++) {
          const double uinst_i = d_inst[n + i * dof];
          const double umean_i = d_mean[n + i * dof];
          delta_i = uinst_i - umean_i;
          for (int j = d_rms_start + 1; j < d_rms_start + d_rms_components; j++) {
            const double uinst_j = d_inst[n + j * dof];
            const double umean_j = d_mean[n + j * dof];
            delta_j = uinst_j - umean_j;

            val = d_vari[n + rms_index * dof];
            d_vari[n + rms_index * dof] = (val * dSamplesRMS + delta_i * delta_j) / (dSamplesRMS + 1);
            rms_index++;
          }
        }
      }  // end variance
    });
  }
}

void Averaging::addSampleInternal(GasMixture *mixture) {
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
    double dSamplesMean = (double)ns_mean_;
    double dSamplesRMS = (double)ns_vari_;

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
