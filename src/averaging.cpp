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
#include "averaging.hpp"

#include <mfem/general/forall.hpp>

#include "dataStructures.hpp"
#include "equation_of_state.hpp"
#include "run_configuration.hpp"
#include "tps.hpp"

AveragingOptions::AveragingOptions() {
  sample_interval_ = 0;
  step_start_mean_ = 0;

  save_mean_history_ = false;
  enable_mean_continuation_ = false;
  zero_variances_ = false;
}

void AveragingOptions::read(TPS::Tps *tps, std::string prefix) {
  std::string basename;
  if (!prefix.empty()) {
    basename = prefix + "/averaging";
  } else {
    basename = "averaging";
  }
  tps->getInput((basename + "/sampleFreq").c_str(), sample_interval_, 0);
  tps->getInput((basename + "/startIter").c_str(), step_start_mean_, 0);
  tps->getInput((basename + "/enableContinuation").c_str(), enable_mean_continuation_, false);
  tps->getInput((basename + "/restartRMS").c_str(), zero_variances_, false);
  tps->getInput((basename + "/saveMeanHist").c_str(), save_mean_history_, false);
}

Averaging::Averaging(AveragingOptions &opts, std::string output_name) {
  rank0_ = false;
  compute_mean_ = false;

  sample_interval_ = opts.sample_interval_;
  step_start_mean_ = opts.step_start_mean_;

  if (sample_interval_ != 0) compute_mean_ = true;

  ns_mean_ = 0;
  ns_vari_ = 0;

  mean_output_name_ = "mean_";
  mean_output_name_.append(output_name);
}

Averaging::~Averaging() {
  if (compute_mean_) {
    delete pvdc_;
    delete meanP;
    delete meanV;
    delete meanRho;
  }
}

void Averaging::registerField(std::string name, const ParGridFunction *field_to_average, bool compute_vari,
                              int vari_start_index, int vari_components) {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  // otherwise, set up ParGridFunction to hold mean...
  ParMesh *mesh = field_to_average->ParFESpace()->GetParMesh();
  rank0_ = (mesh->GetMyRank() == 0);

  ParGridFunction *mean = new ParGridFunction(field_to_average->ParFESpace());
  *mean = 0.0;

  // and maybe the vari
  ParGridFunction *vari = nullptr;
  if (compute_vari) {
    // make sure incoming field has enough components to satisfy vari request
    assert((vari_start_index + vari_components) <= field_to_average->ParFESpace()->GetVDim());

    const int num_variance = vari_components * (vari_components + 1) / 2;

    const FiniteElementCollection *fec = field_to_average->ParFESpace()->FEColl();
    const int order = fec->GetOrder();

    FiniteElementCollection *vari_fec = fec->Clone(order);
    ParFiniteElementSpace *vari_fes = new ParFiniteElementSpace(mesh, vari_fec, num_variance, Ordering::byNODES);
    vari = new ParGridFunction(vari_fes);
    vari->MakeOwner(vari_fec);

    *vari = 0.0;
  }

  // and store those fields in an AveragingFamily object that gets appended to the avg_families_ vector
  avg_families_.emplace_back(AveragingFamily(name, field_to_average, mean, vari, vari_start_index, vari_components));
  std::cout << "Fam size: " << avg_families_.size() << " with addition of " << name << endl;
}

void Averaging::initializeViz() {
  // quick return if not computing stats...
  if (!compute_mean_) return;

  assert(avg_families_.size() >= 1);

  // Loop through the families and add them to the paraview output
  for (size_t i = 0; i < avg_families_.size(); i++) {
    ParGridFunction *mean = avg_families_[i].mean_fcn_;
    ParGridFunction *vari = avg_families_[i].vari_fcn_;

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
  ParGridFunction *vari = avg_families_[0].vari_fcn_;

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
  pvdc_->RegisterField("rms", vari);
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
        if(avg_families_[ifam].vari_fcn_ != nullptr) {	
          *avg_families_[ifam].vari_fcn_ = 0.0;
	}
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
    //std::cout << "starting ifam loop... " << ifam+1 << " of " << avg_families_.size() << " " << avg_families_[ifam].name_ << endl;
    // Extract fields for use on device (when available)
    AveragingFamily &fam = avg_families_[ifam];
    //std::cout << "okay a... " << endl;        

    const ParGridFunction *inst = fam.instantaneous_fcn_;
    ParGridFunction *mean = fam.mean_fcn_;
    ParGridFunction *vari = fam.vari_fcn_;
    //std::cout << "okay b... " << endl;
    if(fam.mean_fcn_ == nullptr) {
      std::cout << "fam.mean_fcn_ is a nullptr " << endl; 
    }

    const double *d_inst = inst->Read();
    //std::cout << "okay b1... " << endl;    
    double *d_mean = mean->ReadWrite();    
    //std::cout << "okay b3... " << endl;        
    double *d_vari;
    //std::cout << "okay b4... " << endl;        
    if(vari != nullptr) { d_vari = vari->ReadWrite(); }
    //std::cout << "okay c... " << endl;            

    // Extract size information for use on device
    const int dof = mean->ParFESpace()->GetNDofs();  // dofs per scalar field
    const int neq = mean->ParFESpace()->GetVDim();   // number of scalar variables in mean field
    //std::cout << "okay d... " << endl;            

    int d_vari_start;
    int d_vari_components;
    if(vari != nullptr) {
      d_vari_start = fam.vari_start_index_;
      d_vari_components = fam.vari_components_;      
    }
    //std::cout << "okay e... " << endl;            

    // Extract sample size information for use on device
    double d_ns_mean = (double)ns_mean_;
    double d_ns_vari;
    if(vari != nullptr) {
      d_ns_vari = (double)ns_vari_;      
    }

    //std::cout << "okay 0... " << dof << endl;    
    // "Loop" over all dofs and update statistics
    MFEM_FORALL(n, dof, {
      // Update mean
      //std::cout << "okay 1..." << endl;
      for (int eq = 0; eq < neq; eq++) {
        const double uinst = d_inst[n + eq * dof];
        const double umean = d_mean[n + eq * dof];
        const double N_umean = d_ns_mean * umean;
        d_mean[n + eq * dof] = (N_umean + uinst) / (d_ns_mean + 1);
      }
      //std::cout << "okay 2..." << endl;      

      // Update variances (only computed if we have a place to put them)
      //if (d_vari != nullptr) {
      if (vari != nullptr) {	
        double val = 0.;
        double delta_i = 0.;
        double delta_j = 0.;
        int vari_index = 0;
        //std::cout << "okay 3..." << endl;      	

        // Variances first (i.e., diagonal of the covariance matrix)
        for (int i = d_vari_start; i < d_vari_start + d_vari_components; i++) {
          const double uinst = d_inst[n + i * dof];
          const double umean = d_mean[n + i * dof];
          delta_i = (uinst - umean);
          val = d_vari[n + vari_index * dof];
          d_vari[n + vari_index * dof] = (val * d_ns_vari + delta_i * delta_i) / (d_ns_vari + 1);
          vari_index++;
        }
        //std::cout << "okay 4... " << d_vari_start << " " << d_vari_components << endl;

        // Covariances second (i.e., off-diagonal components, if any)
        for (int i = d_vari_start; i < d_vari_start + d_vari_components - 1; i++) {
          const double uinst_i = d_inst[n + i * dof];
          const double umean_i = d_mean[n + i * dof];
          delta_i = uinst_i - umean_i;
	  //std::cout << "okay 4a... " << endl;	  
          for (int j = d_vari_start + 1; j < d_vari_start + d_vari_components; j++) {
            const double uinst_j = d_inst[n + j * dof];
            const double umean_j = d_mean[n + j * dof];
            delta_j = uinst_j - umean_j;
  	    //std::cout << "okay 4b... " << endl;	  	    

            val = d_vari[n + vari_index * dof];
  	    //std::cout << "okay 4c... " << endl;	  	    	    
            d_vari[n + vari_index * dof] = (val * d_ns_vari + delta_i * delta_j) / (d_ns_vari + 1);
  	    //std::cout << "okay 4d... " << endl;	  	    	    
            vari_index++;
  	    //std::cout << "okay 4e... " << endl;	  	    	    
          }
        }
      }  // end variance
      //std::cout << "okay 5..." << endl; 
      
    });
    //std::cout << "okay 6..." << endl;     
  }
  //std::cout << "okay 7..." << endl;   
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
    ParGridFunction *vari = fam.vari_fcn_;

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

    const int d_vari_start = fam.vari_start_index_;
    const int d_vari_components = fam.vari_components_;

    // Extract sample size information for use on device
    double d_ns_mean = (double)ns_mean_;
    double d_ns_vari = (double)ns_vari_;

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
        const double N_umean = d_ns_mean * umean;

        if (eq != 1 + dim) {
          d_mean[n + eq * dof] = (N_umean + uinst) / (d_ns_mean + 1);
        } else {  // eq == 1+dim
          double p = nUp[eq];
          if (mixture != nullptr) {
            p = d_mixture->ComputePressureFromPrimitives(nUp);
          }
          d_mean[n + eq * dof] = (N_umean + p) / (d_ns_mean + 1);
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
        int vari_index = 0;

        // Variances first (i.e., diagonal of the covariance matrix)
        for (int i = d_vari_start; i < d_vari_start + d_vari_components; i++) {
          val = d_vari[n + vari_index * dof];
          delta_i = (nUp[i] - mean_state[i]);
          d_vari[n + vari_index * dof] = (val * d_ns_vari + delta_i * delta_i) / (d_ns_vari + 1);
          vari_index++;
        }

        // Covariances second (i.e., off-diagonal components, if any)
        for (int i = d_vari_start; i < d_vari_start + d_vari_components - 1; i++) {
          for (int j = d_vari_start + 1; j < d_vari_start + d_vari_components; j++) {
            val = d_vari[n + vari_index * dof];
            delta_i = (nUp[i] - mean_state[i]);
            delta_j = (nUp[j] - mean_state[j]);
            d_vari[n + vari_index * dof] = (val * d_ns_vari + delta_i * delta_j) / (d_ns_vari + 1);
            vari_index++;
          }
        }
      }  // end variance
    });
  }
}
