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
#include "averaging_and_rms.hpp"

Averaging::Averaging(ParGridFunction *_Up, ParMesh *_mesh, FiniteElementCollection *_fec, ParFiniteElementSpace *_fes,
                     ParFiniteElementSpace *_dfes, ParFiniteElementSpace *_vfes, const int &_num_equation,
                     const int &_dim, RunConfiguration &_config, MPI_Groups *_groupsMPI)
    : Up(_Up),
      mesh(_mesh),
      fec(_fec),
      fes(_fes),
      dfes(_dfes),
      vfes(_vfes),
      num_equation(_num_equation),
      dim(_dim),
      config(_config),
      groupsMPI(_groupsMPI) {
  // Always assume 6 components of the Reynolds stress tensor
  numRMS = 6;

  sampleInterval = config.GetMeanSampleInterval();
  samplesMean = 0;
  startMean = config.GetMeanStartIter();
  computeMean = false;
  if (sampleInterval != 0) computeMean = true;

  if (computeMean) {
    rmsFes = new ParFiniteElementSpace(mesh, fec, numRMS, Ordering::byNODES);

    meanUp = new ParGridFunction(vfes);
    rms = new ParGridFunction(rmsFes);

    meanRho = new ParGridFunction(fes, meanUp->GetData());
    meanV = new ParGridFunction(dfes, meanUp->GetData() + fes->GetNDofs());
    meanP = new ParGridFunction(fes, meanUp->GetData() + (num_equation - 1) * fes->GetNDofs());

    initiMeanAndRMS();

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

    local_sums.UseDevice(true);
    local_sums.SetSize(5 + 6);
    local_sums = 0.;

    tmp_vector.UseDevice(true);
    tmp_vector.SetSize(rms->Size());
    tmp_vector = 0.;
  }
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

void Averaging::addSampleMean(const int &iter) {
  if (computeMean) {
    if (iter % sampleInterval == 0 && iter >= startMean) {
      if (iter == startMean && groupsMPI->getSession()->Root()) cout << "Starting mean calculation." << endl;

      double *dataUp = Up->GetData();
      double *dataMean = meanUp->GetData();
      double *dataRMS = rms->GetData();
      int dof = fes->GetNDofs();

      for (int n = 0; n < dof; n++) {
        // mean
        for (int eq = 0; eq < num_equation; eq++) {
          double mVal = static_cast<double>(samplesMean) * dataMean[n + eq * dof];
          dataMean[n + eq * dof] = (mVal + dataUp[n + eq * dof]) / static_cast<double>(samplesMean + 1);
        }

        // ----- RMS -----
        // velocities
        Vector meanVel(3), vel(3);
        meanVel = 0.;
        vel = 0.;
        for (int d = 0; d < dim; d++) {
          meanVel[d] = dataMean[n + dof * (d + 1)];
          vel[d] = dataUp[n + dof * (d + 1)];
        }

        // xx
        double val = dataRMS[n];
        dataRMS[n] = (val * static_cast<double>(samplesMean) + (vel[0] - meanVel[0]) * (vel[0] - meanVel[0])) /
                     static_cast<double>(samplesMean + 1);
        // yy
        val = dataRMS[n + dof];
        dataRMS[n + dof] = (val * static_cast<double>(samplesMean) + (vel[1] - meanVel[1]) * (vel[1] - meanVel[1])) /
                           static_cast<double>(samplesMean + 1);
        // zz
        val = dataRMS[n + 2 * dof];
        dataRMS[n + 2 * dof] =
            (val * static_cast<double>(samplesMean) + (vel[2] - meanVel[2]) * (vel[2] - meanVel[2])) /
            static_cast<double>(samplesMean + 1);
        // xy
        val = dataRMS[n + 3 * dof];
        dataRMS[n + 3 * dof] =
            (val * static_cast<double>(samplesMean) + (vel[0] - meanVel[0]) * (vel[1] - meanVel[1])) /
            static_cast<double>(samplesMean + 1);
        // xz
        val = dataRMS[n + 4 * dof];
        dataRMS[n + 4 * dof] =
            (val * static_cast<double>(samplesMean) + (vel[0] - meanVel[0]) * (vel[2] - meanVel[2])) /
            static_cast<double>(samplesMean + 1);
        // yz
        val = dataRMS[n + 5 * dof];
        dataRMS[n + 5 * dof] =
            (val * static_cast<double>(samplesMean) + (vel[1] - meanVel[1]) * (vel[2] - meanVel[2])) /
            static_cast<double>(samplesMean + 1);
      }

      samplesMean++;
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

void Averaging::read_meanANDrms_restart_files() {
  //   if( computeMean && config.GetRestartMean() )
  //   {
  //     string serialName = "restart_mean_";
  //     serialName.append( config.GetOutputName() );
  //     serialName.append( ".sol" );
  //
  //     string serialNameRMS = "restart_rms_";
  //     serialNameRMS.append( config.GetOutputName() );
  //     serialNameRMS.append( ".sol" );
  //
  //     string fileName = groupsMPI->getParallelName( serialName );
  //     ifstream file( fileName );
  //
  //     if( !file.is_open() )
  //     {
  //       cout<< "Could not open file \""<<fileName<<"\""<<endl;
  //       return;
  //     }else
  //     {
  //       double *dataUp = meanUp->GetData();
  //
  //       string line;
  //       // read time and iters
  //       {
  //         getline(file,line);
  //         istringstream ss(line);
  //         string word;
  //         ss >> word;
  //         samplesMean = stoi( word );
  //
  //         ss >> word;
  //         sampleInterval = stof( word );
  //       }
  //
  //       int lines = 0;
  //       while( getline(file,line) )
  //       {
  //         istringstream ss(line);
  //         string word;
  //         ss >> word;
  //
  //         dataUp[lines] = stof( word );
  //         lines++;
  //       }
  //       file.close();
  //     }
  //
  //     // read RMS
  //     string rmsName  = groupsMPI->getParallelName( serialNameRMS );
  //     ifstream rmsfile( rmsName );
  //
  //     if( !rmsfile.is_open() )
  //     {
  //       cout<< "Could not open file \""<<rmsName<<"\""<<endl;
  //       return;
  //     }else
  //     {
  //       double *dataRMS = rms->GetData();
  //
  //       string line;
  //
  //       int lines = 0;
  //       while( getline(rmsfile,line) )
  //       {
  //         istringstream ss(line);
  //         string word;
  //         ss >> word;
  //
  //         dataRMS[lines] = stof( word );
  //         lines++;
  //       }
  //       rmsfile.close();
  //     }
  //   }
}

void Averaging::initiMeanAndRMS() {
  double *dataMean = meanUp->GetData();
  double *dataRMS = rms->GetData();
  int dof = vfes->GetNDofs();

  for (int i = 0; i < dof; i++) {
    for (int eq = 0; eq < num_equation; eq++) dataMean[i + eq * dof] = 0.;
    for (int n = 0; n < numRMS; n++) dataRMS[i + dof * n] = 0.;
  }
}

const double *Averaging::getLocalSums() {
#ifdef _GPU_
  sumValues_gpu(*meanUp, *rms, local_sums, tmp_vector, num_equation, dim);
#else
  const int NDof = meanUp->Size() / num_equation;
  const double ddof = static_cast<double>NDof;

  double *dataMean = meanUp->GetData();
  double *dataRMS = rms->GetData();

  local_sums = 0.;
  for (int n = 0; n < NDof; n++) {
    for (int eq = 0; eq < num_equation; eq++) local_sums(eq) += fabs(dataMean[n + eq * NDof]) / ddof;
    for (int i = 0; i < 6; i++) local_sums(5 + i) += fabs(dataRMS[n + i * NDof]) / ddof;
  }
  if (dim == 2) {
    local_sums(4) = local_sums(3);
    local_sums(3) = 0.;
  }
#endif

  return local_sums.HostRead();
}

#ifdef _GPU_
void Averaging::sumValues_gpu(const Vector &meanUp, const Vector &rms, Vector &local_sums, Vector &tmp_vector,
                              const int &num_equation, const int &dim) {
  const int NDof = meanUp.Size() / num_equation;

  auto d_Up = meanUp.Read();
  auto d_rms = rms.Read();
  auto d_tmp = tmp_vector.Write();
  auto d_loc = local_sums.Write();

  // copy up values to temp vector
  MFEM_FORALL(n, meanUp.Size(), { d_tmp[n] = fabs(d_Up[n]); });

  // sum up all Up values
  MFEM_FORALL(n, NDof, {
    int interval = 1;
    while (interval < NDof) {
      interval *= 2;
      if (n % interval == 0) {
        int n2 = n + interval / 2;
        for (int eq = 0; eq < num_equation; eq++) {
          if (n2 < NDof) d_tmp[n + eq * NDof] += d_tmp[n2 + eq * NDof];
        }
      }
      MFEM_SYNC_THREAD;
    }
  });

  // transfer to smaller vector
  MFEM_FORALL(eq, num_equation, {
    double ddof = (double)NDof;
    d_loc[eq] = d_tmp[eq * NDof] / ddof;
  });

  // rearrange
  if (dim == 2) {
    MFEM_FORALL(eq, num_equation, {
      if (eq == 0) {
        d_loc[4] = d_loc[3];
        d_loc[3] = 0.;
      }
    });
  }

  // REPEAT FOR RMS
  MFEM_FORALL(n, rms.Size(), { d_tmp[n] = fabs(d_rms[n]); });

  MFEM_FORALL(n, NDof, {
    int interval = 1;
    while (interval < NDof) {
      interval *= 2;
      if (n % interval == 0) {
        int n2 = n + interval / 2;
        for (int eq = 0; eq < 6; eq++) {
          if (n2 < NDof) d_tmp[n + eq * NDof] += d_tmp[n2 + eq * NDof];
        }
      }
      MFEM_SYNC_THREAD;
    }
  });

  MFEM_FORALL(eq, 6, {
    double ddof = (double)NDof;
    d_loc[5 + eq] = d_tmp[eq * NDof] / ddof;
  });
}

#endif  // _GPU_
