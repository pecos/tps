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
Averaging::Averaging(ParGridFunction *_Up, ParMesh *_mesh, FiniteElementCollection *_fec, ParFiniteElementSpace *_fes,
                     ParFiniteElementSpace *_dfes, ParFiniteElementSpace *_vfes, Equations &_eqSys,
                     GasMixture *_mixture, const int &_num_equation, const int &_dim, RunConfiguration &_config,
                     MPI_Groups *_groupsMPI)
    : Up(_Up),
      mesh(_mesh),
      fec(_fec),
      fes(_fes),
      dfes(_dfes),
      vfes(_vfes),
      eqSystem(_eqSys),
      num_equation(_num_equation),
      dim(_dim),
      nvel(_config.isAxisymmetric() ? 3 : _dim),
      config(_config),
      groupsMPI(_groupsMPI),
      mixture(_mixture) {
  // Always assume 6 components of the Reynolds stress tensor
  numRMS = 6;

  sampleInterval = config.GetMeanSampleInterval();
  samplesMean = 0;
  startMean = config.GetMeanStartIter();
  computeMean = false;
  if (sampleInterval != 0) computeMean = true;

  if (computeMean) {
    //     mixture = new DryAir(config,dim);

    rmsFes = new ParFiniteElementSpace(mesh, fec, numRMS, Ordering::byNODES);

    meanUp = new ParGridFunction(vfes);
    rms = new ParGridFunction(rmsFes);

    meanRho = new ParGridFunction(fes, meanUp->GetData());
    meanV = new ParGridFunction(dfes, meanUp->GetData() + fes->GetNDofs());
    meanP = new ParGridFunction(fes, meanUp->GetData() + (1 + nvel) * fes->GetNDofs());

    meanScalar = NULL;
    if (eqSystem == NS_PASSIVE)
      meanScalar = new ParGridFunction(fes, meanUp->GetData() + (num_equation - 1) * fes->GetNDofs());

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
    if (eqSystem == NS_PASSIVE) paraviewMean->RegisterField("passScalar", meanScalar);

    // NOTE(kevin): this variable is currently obsolete.
    // It represents `dof`-averaged state, which is useless at this point.
    // This will not be supported.
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
    //     delete mixture;
    delete paraviewMean;

    delete meanP;
    delete meanV;
    delete meanRho;
    if (eqSystem == NS_PASSIVE) delete meanScalar;

    delete rms;
    delete meanUp;

    delete rmsFes;
  }
}

void Averaging::addSampleMean(const int &iter) {
  if (computeMean) {
    if (iter % sampleInterval == 0 && iter >= startMean) {
      if (iter == startMean && groupsMPI->isWorldRoot()) cout << "Starting mean calculation." << endl;
#ifdef _GPU_
      addSample_gpu(meanUp, rms, samplesMean, mixture, Up, fes->GetNDofs(), dim, num_equation);
#else
      addSample_cpu();
#endif
      samplesMean++;
    }
  }
}

void Averaging::addSample_cpu() {
  double *dataUp = Up->GetData();
  double *dataMean = meanUp->GetData();
  double *dataRMS = rms->GetData();
  int dof = fes->GetNDofs();

  Vector iUp(num_equation);

  for (int n = 0; n < dof; n++) {
    for (int eq = 0; eq < num_equation; eq++) iUp[eq] = dataUp[n + eq * dof];

    // mean
    for (int eq = 0; eq < num_equation; eq++) {
      double mVal = double(samplesMean) * dataMean[n + eq * dof];
      dataMean[n + eq * dof] = (mVal + iUp[eq]) / double(samplesMean + 1);

      // presseure-temperature change
      if (eq == dim + 1) {
        double p = mixture->ComputePressureFromPrimitives(iUp);
        dataMean[n + eq * dof] = (mVal + p) / double(samplesMean + 1);
      }
    }

    // ----- RMS -----
    // velocities
    Vector meanVel(3), vel(3);
    meanVel = 0.;
    vel = 0.;
    for (int d = 0; d < nvel; d++) {
      meanVel[d] = dataMean[n + dof * (d + 1)];
      vel[d] = dataUp[n + dof * (d + 1)];
    }

    // xx
    double val = dataRMS[n];
    dataRMS[n] = (val * double(samplesMean) + (vel[0] - meanVel[0]) * (vel[0] - meanVel[0])) / double(samplesMean + 1);
    // yy
    val = dataRMS[n + dof];
    dataRMS[n + dof] =
        (val * double(samplesMean) + (vel[1] - meanVel[1]) * (vel[1] - meanVel[1])) / double(samplesMean + 1);
    // zz
    val = dataRMS[n + 2 * dof];
    dataRMS[n + 2 * dof] =
        (val * double(samplesMean) + (vel[2] - meanVel[2]) * (vel[2] - meanVel[2])) / double(samplesMean + 1);
    // xy
    val = dataRMS[n + 3 * dof];
    dataRMS[n + 3 * dof] =
        (val * double(samplesMean) + (vel[0] - meanVel[0]) * (vel[1] - meanVel[1])) / double(samplesMean + 1);
    // xz
    val = dataRMS[n + 4 * dof];
    dataRMS[n + 4 * dof] =
        (val * double(samplesMean) + (vel[0] - meanVel[0]) * (vel[2] - meanVel[2])) / double(samplesMean + 1);
    // yz
    val = dataRMS[n + 5 * dof];
    dataRMS[n + 5 * dof] =
        (val * double(samplesMean) + (vel[1] - meanVel[1]) * (vel[2] - meanVel[2])) / double(samplesMean + 1);
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
  double *dataMean = meanUp->HostWrite();
  double *dataRMS = rms->HostWrite();
  int dof = vfes->GetNDofs();

  for (int i = 0; i < dof; i++) {
    for (int eq = 0; eq < num_equation; eq++) dataMean[i + eq * dof] = 0.;
    for (int n = 0; n < numRMS; n++) dataRMS[i + dof * n] = 0.;
  }
}

// NOTE(kevin): this routine is currently obsolete.
// It computes `dof`-averaged state and time-derivative, which are useless at this point.
// This will not be supported.
const double *Averaging::getLocalSums() {
#ifdef _GPU_
  sumValues_gpu(*meanUp, *rms, local_sums, tmp_vector, num_equation, dim);
#else
  const int NDof = meanUp->Size() / num_equation;
  const double ddof = static_cast<double>(NDof);

  double *dataMean = meanUp->GetData();
  double *dataRMS = rms->GetData();

  local_sums = 0.;
  for (int n = 0; n < NDof; n++) {
    for (int eq = 0; eq < num_equation; eq++) local_sums(eq) += fabs(dataMean[n + eq * NDof]) / ddof;
    for (int i = 0; i < 6; i++) local_sums(5 + i) += fabs(dataRMS[n + i * NDof]) / ddof;
  }
  if (nvel == 2) {
    local_sums(4) = local_sums(3);
    local_sums(3) = 0.;
  }
#endif

  return local_sums.HostRead();
}

#ifdef _GPU_
void Averaging::addSample_gpu(ParGridFunction *meanUp, ParGridFunction *rms, int &samplesMean, GasMixture *mixture,
                              const ParGridFunction *Up, const int &Ndof, const int &dim, const int &num_equation) {
  double *d_meanUp = meanUp->ReadWrite();
  double *d_rms = rms->ReadWrite();
  const double *d_Up = Up->Read();

  double dSamplesMean = (double)samplesMean;

  GasMixture *d_mixture = mixture;

  MFEM_FORALL(n, Ndof, {
    double meanVel[gpudata::MAXDIM], vel[gpudata::MAXDIM];  // double meanVel[3], vel[3];
    // double nUp[20];  // NOTE: lets make sure we don't have more than 20 eq.
    // NOTE(kevin): (presumably) marc left this hidden note here..
    double nUp[gpudata::MAXEQUATIONS];

    for (int eq = 0; eq < num_equation; eq++) {
      nUp[eq] = d_Up[n + eq * Ndof];
    }

    // mean
    for (int eq = 0; eq < num_equation; eq++) {
      double mUpi = d_meanUp[n + eq * Ndof];
      double mVal = dSamplesMean * mUpi;

      double newMeanUp;
      if (eq != 1 + dim) {
        newMeanUp = (mVal + nUp[eq]) / (dSamplesMean + 1);
      } else {  // eq == 1+dim
        double p = d_mixture->ComputePressureFromPrimitives(nUp);
        newMeanUp = (mVal + p) / (dSamplesMean + 1);
      }

      d_meanUp[n + eq * Ndof] = newMeanUp;
    }

    // fill out mean velocity array
    for (int d = 0; d < dim; d++) {
      meanVel[d] = d_meanUp[n + (d + 1) * Ndof];
      vel[d] = nUp[d + 1];
    }
    if (dim != 3) {
      meanVel[2] = 0.;
      vel[2] = 0.;
    }

    // ----- RMS -----
    double val = 0.;
    // xx
    val = d_rms[n];
    d_rms[n] = (val * dSamplesMean + (vel[0] - meanVel[0]) * (vel[0] - meanVel[0])) / (dSamplesMean + 1);

    // yy
    val = d_rms[n + Ndof];
    d_rms[n + Ndof] = (val * dSamplesMean + (vel[1] - meanVel[1]) * (vel[1] - meanVel[1])) / (dSamplesMean + 1);

    // zz
    val = d_rms[n + 2 * Ndof];
    d_rms[n + 2 * Ndof] = (val * dSamplesMean + (vel[2] - meanVel[2]) * (vel[2] - meanVel[2])) / (dSamplesMean + 1);

    // xy
    val = d_rms[n + 3 * Ndof];
    d_rms[n + 3 * Ndof] = (val * dSamplesMean + (vel[0] - meanVel[0]) * (vel[1] - meanVel[1])) / (dSamplesMean + 1);

    // xz
    val = d_rms[n + 4 * Ndof];
    d_rms[n + 4 * Ndof] = (val * dSamplesMean + (vel[0] - meanVel[0]) * (vel[2] - meanVel[2])) / (dSamplesMean + 1);

    // yz
    val = d_rms[n + 5 * Ndof];
    d_rms[n + 5 * Ndof] = (val * dSamplesMean + (vel[1] - meanVel[1]) * (vel[2] - meanVel[2])) / (dSamplesMean + 1);
  });
}

// NOTE(kevin): this routine is currently obsolete.
// It computes `dof`-averaged state and time-derivative, which are useless at this point.
// This will not be supported.
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
