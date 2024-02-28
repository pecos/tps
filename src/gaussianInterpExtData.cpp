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

#include "gaussianInterpExtData.hpp"

#include <hdf5.h>

#include <fstream>
#include <iomanip>

#include "../utils/mfem_extras/pfem_extras.hpp"
#include "loMach.hpp"
#include "loMach_options.hpp"
#include "logger.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/linalg/solvers.hpp"
#include "externalData_base.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace mfem::common;

GaussianInterpExtData::GaussianInterpExtData(mfem::ParMesh *pmesh, LoMachOptions *loMach_opts, TPS::Tps *tps)
    : tpsP_(tps),
      groupsMPI(new MPI_Groups(tps->getTPSCommWorld())),
      nprocs_(groupsMPI->getTPSWorldSize()),
      pmesh_(pmesh),
      loMach_opts_(loMach_opts) {
  rank_ = pmesh_->GetMyRank();
  rank0_ = (pmesh_->GetMyRank() == 0);
  order_ = loMach_opts->order;

  // only allows one inlet now
  isInterpInlet_ = false;  
  std::string type;
  std::string basepath("boundaryConditions/inlet1");
  tpsP_->getRequiredInput((basepath + "/type").c_str(), type);
  if (type == "interpolate") {
    isInterpInlet_ = true;
    tpsP_->getInput((basepath + "/name").c_str(), fname_, std::string("inletPlane.csv"));    
  }  

  // TODO: add checks and error msg for invalid sponge settings
}

void GaussianInterpExtData::initializeSelf() {
  rank_ = pmesh_->GetMyRank();
  rank0_ = false;
  if (rank_ == 0) {
    rank0_ = true;
  }
  dim_ = pmesh_->Dimension();
  nvel_ = dim_;

  if(!isInterpInlet_) {return;}
  
  bool verbose = rank0_;
  if (verbose) grvy_printf(ginfo, "Initializing Gaussian interpolation inlet.\n");

  sfec_ = new H1_FECollection(order_);
  sfes_ = new ParFiniteElementSpace(pmesh_, sfec_);

  vfec_ = new H1_FECollection(order_);
  vfes_ = new ParFiniteElementSpace(pmesh_, vfec_, dim_);

  Sdof_ = sfes_->GetNDofs();
  // int sfes_truevsize = sfes_->GetTrueVSize();
  
  temperature_gf_.SetSpace(sfes_);
  velocityU_gf_.SetSpace(sfes_);
  velocityV_gf_.SetSpace(sfes_);
  velocityW_gf_.SetSpace(sfes_);    

  // exports
  toThermoChem_interface_.Tdata = &temperature_gf_;
  toFlow_interface_.Udata = &velocityU_gf_;
  toFlow_interface_.Vdata = &velocityV_gf_;
  toFlow_interface_.Wdata = &velocityW_gf_;  
}

void GaussianInterpExtData::initializeViz(ParaViewDataCollection &pvdc) {
  if(!isInterpInlet_) {return;}  
  pvdc.RegisterField("externalTemp", &temperature_gf_);
  pvdc.RegisterField("externalU", &velocityU_gf_);
  pvdc.RegisterField("externalV", &velocityV_gf_);
  pvdc.RegisterField("externalW", &velocityW_gf_);
}

// TODO: add a translation and rotation for external data plane
void GaussianInterpExtData::setup() {

  if(!isInterpInlet_) {return;}  
  ParGridFunction coordsDof(vfes_);
  pmesh_->GetNodes(coordsDof);

  double *Tdata = temperature_gf_.HostReadWrite();
  double *Udata = velocityU_gf_.HostReadWrite();
  double *Vdata = velocityV_gf_.HostReadWrite();
  double *Wdata = velocityW_gf_.HostReadWrite();    
  double *hcoords = coordsDof.HostReadWrite();

  struct inlet_profile {
    double x, y, z, rho, temp, u, v, w;
  };  
  
  // string fname;
  // fname = "./inputs/inletPlane.csv";
  // const int fname_length = fname.length();
  // char* char_array = new char[fname_length + 1];
  // strcpy(char_array, fname.c_str());

  string fnameBase;  
  std::string basepath("./inputs/");
  fnameBase = (basepath + fname_);
  const char* fnameRead = fnameBase.c_str(); 
  
  int nCount = 0;
    
  // find size    
  if(rank0_) {

    std::cout << " Attempting to open inlet file for counting... " << fnameRead << endl; fflush(stdout);

      // HARDCODE
      FILE *inlet_file;
      // if ( inlet_file = fopen("./inputs/inletPlane.csv","r") ) {
      if ( inlet_file = fopen(fnameRead,"r") ) {	
	std::cout << " ...and open" << endl; fflush(stdout);
      }
      else {
        std::cout << " ...CANNOT OPEN FILE" << endl; fflush(stdout);
      }
      
      char* line = NULL;
      int ch = 0;
      size_t len = 0;
      ssize_t read;

      std::cout << " ...starting count" << endl; fflush(stdout);
      for (ch = getc(inlet_file); ch != EOF; ch = getc(inlet_file)) {
        if (ch == '\n') {
          nCount++;
	}
      }      
      fclose(inlet_file);      
      std::cout << " final count: " << nCount << endl; fflush(stdout);

    }

    // broadcast size
    MPI_Bcast(&nCount, 1, MPI_INT, 0, tpsP_->getTPSCommWorld());

    struct inlet_profile inlet[nCount];
    double junk;    
    
    // mean output from plane interpolation
    // 0) no, 1) x, 2) y, 3) z, 4) rho, 5) u, 6) v, 7) w
    // Change format so temp is given instead of rho!
    
    // open, read data
    if (rank0_) {    
    
      ifstream file;

      // HARDCODE
      file.open("./inputs/inletPlane.csv");
      
      string line;
      getline(file, line);
      int nLines = 0;
      while (getline(file, line)) {
        // cout << line << endl;
	stringstream sline(line);
	int entry = 0;
        while (sline.good()) {
          string substr;
          getline(sline, substr, ','); // csv delimited
          double buffer = std::stod(substr);
	  entry++;

	  // using code plane interp format
	  if (entry == 2) {
	    inlet[nLines].x = buffer;
	  } else if (entry == 3) {
	    inlet[nLines].y = buffer;
	  } else if (entry == 4) {
	    inlet[nLines].z = buffer;
	  } else if (entry == 5) {	    
	    // inlet[nLines].rho = buffer;
	    // to prevent nans in temp in out-of-domain plane regions
	    // inlet[nLines].temp = thermoPressure / (Rgas * std::max(buffer,1.0));

	    // just change format so 5 slot has temp
	    inlet[nLines].temp = buffer;
	    
	  } else if (entry == 6) {
	    inlet[nLines].u = buffer;	    
	  } else if (entry == 7) {
	    inlet[nLines].v = buffer;
	  } else if (entry == 8) {
	    inlet[nLines].w = buffer;
	  }
	  
        }
	nLines++;
      }
      file.close();
      std::cout << " " << endl; fflush(stdout);    
    }
  
    // broadcast data
    MPI_Bcast(&inlet, nCount*8, MPI_DOUBLE, 0, tpsP_->getTPSCommWorld());
    if(rank0_) { std::cout << " communicated inlet data" << endl; fflush(stdout); }

    /*
    // width of interpolation stencil
    double Lx, Ly, Lz, Lmax;
    Lx = xmax - xmin;
    Ly = ymax - ymin;
    Lz = zmax - zmin;
    Lmax = std::max(Lx,Ly);
    Lmax = std::max(Lmax,Lz);    
    //radius = 0.25 * Lmax;
    //radius = 0.01 * Lmax;

    double torch_radius = 28.062/1000.0;
    radius = 2.0 * torch_radius / 300.0; // 300 from nxn interp plane
    */
    double radius;
    
    // TODO: dont need to fill entire gf, just appropriate face
    for (int be = 0; be < pmesh_->GetNBE(); be++) {
      int bAttr = pmesh_->GetBdrElement(be)->GetAttribute();
      if (bAttr == InletType::INTERPOLATE) {
        Array<int> vdofs;
        sfes_->GetBdrElementVDofs(be, vdofs);
        for (int i = 0; i < vdofs.Size(); i++) {

	  // index in gf of bndry element
	  int n = vdofs[i];

          // get coords
          auto hcoords = coordsDof.HostRead();  
          double xp[dim_];      
          for (int d = 0; d < dim_; d++) {
            xp[d] = hcoords[n + d * vfes_->GetNDofs()];
          }

          int iCount = 0;      
          double dist, wt;
          double wt_tot = 0.0;          
          // double val_rho = 0.0;
          double val_u = 0.0;
          double val_v = 0.0;
          double val_w = 0.0;
          double val_T = 0.0;	  

          // minimum distance in interpolant field
          double distMin = 1.0e12;
          for (int j = 0; j < nCount; j++) {	    
            dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	         + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	         + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);
            dist = sqrt(dist);
    	    distMin = std::min(distMin, dist);
          }
      
          // find second closest data pt
          double distMinSecond = 1.0e12;
          for (int j = 0; j < nCount; j++) {	    
            dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	         + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	         + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);	
            dist = sqrt(dist);
  	    if (dist > distMin) {
	    distMinSecond = std::min(distMinSecond, dist);
	  }
        }      

        // radius for Gaussian interpolation
        radius = distMinSecond;
        // std::cout << " radius for interpoaltion: " << radius << endl; fflush(stdout);
      
        for (int j=0; j < nCount; j++) {
	    
          dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	       + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	       + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);	
          dist = sqrt(dist);

	  // exclude points outside the domain
	  if (inlet[j].temp < 1.0e-8) { continue; }
	
	  // gaussian interpolation
	  if(dist <= 1.5*radius) {
            // std::cout << " Caught an interp point " << dist << endl; fflush(stdout);	  
            wt = exp(-(dist*dist)/(radius*radius));
            wt_tot = wt_tot + wt;	      
            // val_rho = val_rho + wt*inlet[j].rho;
            val_u = val_u + wt*inlet[j].u;
            val_v = val_v + wt*inlet[j].v;
            val_w = val_w + wt*inlet[j].w;	  
            val_T = val_T + wt*inlet[j].temp;
   	    iCount++;
            // std::cout << "* " << n << " "  << iCount << " " << wt_tot << " " << val_T << endl; fflush(stdout);	  	  
	  }

	  // nearest, just for testing
	  // if(dist <= distMin) {
  	  //  wt_tot = 1.0;
          //   val_rho = inlet[j].rho;
          //   val_u = inlet[j].u;
          //   val_v = inlet[j].v;
          //   val_w = inlet[j].w;
          //   val_T = inlet[j].temp;	
	  //   iCount = 1;
	  // }
	
        }

        if (wt_tot > 0.0) {
          Udata[n] = val_u / wt_tot;
          Vdata[n] = val_v / wt_tot;
          Wdata[n] = val_w / wt_tot;
          Tdata[n] = val_T / wt_tot;
          // std::cout << n << " point set to: " << dataVel[n + 0*Sdof] << " " << dataTemp[n] << endl; fflush(stdout);
        } else {
          Udata[n] = 0.0;
          Vdata[n] = 0.0;
          Wdata[n] = 0.0;
          Tdata[n] = 0.0;
          // std::cout << " iCount of zero..." << endl; fflush(stdout);	  
        }
        	  
	}
      }
    }


    /*
    for (int n = 0; n < sfes->GetNDofs(); n++) {

      // get coords
      auto hcoords = coordsDof.HostRead();  
      double xp[3];      
      for (int d = 0; d < dim; d++) {
        xp[d] = hcoords[n + d * vfes->GetNDofs()];
      }

      int iCount = 0;      
      double dist, wt;
      double wt_tot = 0.0;          
      double val_rho = 0.0;
      double val_u = 0.0;
      double val_v = 0.0;
      double val_w = 0.0;
      double val_T = 0.0;	  
      double dmin = 1.0e15;

      // minimum distance in interpolant field
      double distMin = 1.0e12;
      for (int j = 0; j < nCount; j++) {	    
        dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	     + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	     + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);
        dist = sqrt(dist);
	distMin = std::min(distMin, dist);
      }
      
      // find second closest data pt
      double distMinSecond = 1.0e12;
      for (int j = 0; j < nCount; j++) {	    
        dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	     + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	     + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);	
        dist = sqrt(dist);
	if (dist > distMin) {
	  distMinSecond = std::min(distMinSecond, dist);
	}
      }      

      // radius for Gaussian interpolation
      radius = distMinSecond;
      // std::cout << " radius for interpoaltion: " << radius << endl; fflush(stdout);
      
      for (int j=0; j < nCount; j++) {
	    
        dist = (xp[0]-inlet[j].x)*(xp[0]-inlet[j].x)
   	     + (xp[1]-inlet[j].y)*(xp[1]-inlet[j].y)
	     + (xp[2]-inlet[j].z)*(xp[2]-inlet[j].z);	
        dist = sqrt(dist);

	// exclude points outside the domain
	if (inlet[j].rho < 1.0e-8) { continue; }
	
	// gaussian interpolation
	if(dist <= 1.5*radius) {
          // std::cout << " Caught an interp point " << dist << endl; fflush(stdout);	  
          wt = exp(-(dist*dist)/(radius*radius));
          wt_tot = wt_tot + wt;	      
          val_rho = val_rho + wt*inlet[j].rho;
          val_u = val_u + wt*inlet[j].u;
          val_v = val_v + wt*inlet[j].v;
          val_w = val_w + wt*inlet[j].w;	  
          val_T = val_T + wt*inlet[j].temp;
   	  iCount++;
          // std::cout << "* " << n << " "  << iCount << " " << wt_tot << " " << val_T << endl; fflush(stdout);	  	  
	}

	// nearest, just for testing
	// if(dist <= dmin) {
	//  dmin = dist;
	//  wt_tot = 1.0;
        //   val_rho = inlet[j].rho;
        //   val_u = inlet[j].u;
        //   val_v = inlet[j].v;
        //   val_w = inlet[j].w;
        //   val_T = inlet[j].temp;	
	//   iCount = 1;
	// }
	
      }

      if (wt_tot > 0.0) {
        Udata[n] = val_u / wt_tot;
        Vdata[n] = val_v / wt_tot;
        Wdata[n] = val_w / wt_tot;
        Tdata[n] = val_T / wt_tot;
        // std::cout << n << " point set to: " << dataVel[n + 0*Sdof] << " " << dataTemp[n] << endl; fflush(stdout);
      } else {
        Udata[n] = 0.0;
        Vdata[n] = 0.0;
        Wdata[n] = 0.0;
        Tdata[n] = 0.0;
        // std::cout << " iCount of zero..." << endl; fflush(stdout);	  
      }
      
    }
    */
    
}

void GaussianInterpExtData::step() {
  // empty for now, use for any updates/modifications
  // during simulation e.g. ramping up with time
}
