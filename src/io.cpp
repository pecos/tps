#include "M2ulPhyS.hpp"
#include <hdf5.h>
#include "utils.hpp"

void M2ulPhyS::restart_files_hdf5(string mode)
{
#ifdef HAVE_GRVY
  grvy_timer_begin(__func__);
#endif

  hid_t file = -1;
  hid_t data_soln;
  herr_t status;
  Vector dataSerial;

  string serialName = "restart_";
  serialName.append( config.GetOutputName() );
  serialName.append( ".sol.h5" );
  string fileName;

  if( ((config.RestartSerial() == "read") && (mode == "read")) || (config.RestartSerial() == "write") )
    fileName  = serialName;
  else
    fileName = groupsMPI->getParallelName( serialName );

  // Variables used if (and only if) restarting from different order
  FiniteElementCollection *aux_fec=NULL;
  ParFiniteElementSpace *aux_vfes=NULL;
  ParGridFunction *aux_U=NULL;
  double *aux_U_data=NULL;
  int auxOrder = -1;

  if(mpi.Root())
    cout << "HDF5 restart files mode: " << mode << endl;

  assert( (mode == "read") || (mode == "write") );

  // open restart files (currently per MPI process variants)
  if (mode == "write")
    {
      if (mpi.Root() || (config.RestartSerial() != "write") )
        {
          file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
          assert(file >= 0);
        }
    }
  else if (mode == "read")
    {
      if(config.RestartSerial() == "read")
	{
	  if(rank0_)
	    {
	      file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	      assert(file >= 0);
	    }
	}
      else
	{
	  // verify we have all desired files and open on each process
	  int gstatus;
	  int status = int(file_exists(fileName));
	  MPI_Allreduce(&status,&gstatus,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);

	  if(gstatus == 0)
	    {
	      grvy_printf(ERROR,"[ERROR]: Unable to access desired restart file -> %s\n",fileName.c_str());
	      exit(ERROR);
	    }

	  file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	  assert(file >= 0);
	}
    }

  // -------------------------------------------------------------------
  // Attributes - relevant solution metadata saved as attributes
  // -------------------------------------------------------------------

  hid_t aid, attr;
  if (mode == "write")
    {
      // note: all tasks save unless we are writing a serial restart file
      if (mpi.Root() || (config.RestartSerial() != "write") )
        {
	  // current iteration count
	  h5_save_attribute(file,"iteration",iter);
	  // total time
	  h5_save_attribute(file,"time",time);
	  // timestep
	  h5_save_attribute(file,"dt",dt);
	  // solution order
	  h5_save_attribute(file,"order",order);
	  // spatial dimension
	  h5_save_attribute(file,"dimension",dim);
	  // code revision
#ifdef BUILD_VERSION
          {
            hid_t ctype     = H5Tcopy (H5T_C_S1);
            int shaLength   = strlen(BUILD_VERSION);
            hsize_t dims[1] = {1};
            H5Tset_size(ctype,shaLength);

            hid_t dspace1dim = H5Screate_simple(1,dims,NULL);

            attr = H5Acreate(file,"revision", ctype, dspace1dim, H5P_DEFAULT, H5P_DEFAULT);
            assert(attr >= 0);
            status = H5Awrite(attr,ctype,BUILD_VERSION);
            assert(status >= 0);
            H5Sclose(dspace1dim);
            H5Aclose(attr);
          }
	}
#endif
    }
  else	// read
    {
      int read_order;

      // normal restarts have each process read there own portion of the
      // solution; a serial restart only reads on rank 0 and distributes to
      // the remaining processes

      if (rank0_ || (config.RestartSerial() != "read") )
	{
	  h5_read_attribute(file,"iteration",iter);
	  h5_read_attribute(file,"time",time);
	  h5_read_attribute(file,"dt",dt);
	  h5_read_attribute(file,"order",read_order);
	}

      if(config.RestartSerial() == "read")
	{
	  MPI_Bcast(&iter,      1, MPI_INT,    0, MPI_COMM_WORLD);
	  MPI_Bcast(&time,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&dt,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(&read_order,1, MPI_INT,    0, MPI_COMM_WORLD);
	}

      if( loadFromAuxSol )
        {
	  assert(config.RestartSerial() == "no");
          auxOrder = read_order;

          if( basisType == 0 )
            aux_fec = new DG_FECollection(auxOrder, dim, BasisType::GaussLegendre);
          else if ( basisType == 1 )
            aux_fec = new DG_FECollection(auxOrder, dim, BasisType::GaussLobatto);

          aux_vfes = new ParFiniteElementSpace(mesh, aux_fec, num_equation,
                                               Ordering::byNODES);

          aux_U_data = new double[num_equation*aux_vfes->GetNDofs()];
          aux_U = new ParGridFunction(aux_vfes, aux_U_data);
        }
      else
        assert(read_order==order);

      if(rank0_)
	{
	  grvy_printf(GRVY_INFO,"Restarting from iteration = %i\n",iter);
	  grvy_printf(GRVY_INFO,"--> time = %e\n",time);
	  grvy_printf(GRVY_INFO,"--> dt   = %e\n",dt);
	}

    }

  // -------------------------------------------------------------------
  // Read/write solution data defined by IO families
  // -------------------------------------------------------------------

  hsize_t dims[1];
  hsize_t maxdims[1];
  hsize_t numInSoln;

  if(mode == "write")
    {
      if ( (config.RestartSerial() == "write") && (nprocs_ > 1) && (mpi.Root()) )
	{
	  assert( (locToGlobElem != NULL) && (partitioning_ != NULL) );
          assert( serial_fes != NULL );
	  dims[0] = serial_fes->GetNDofs();
	}
      else
        dims[0] = vfes->GetNDofs();

      //-------------------------------------------------------
      // Loop over defined IO families to save desired output
      //-------------------------------------------------------
      for(auto fam : ioData.families_)
        {
          // define groups based on defined IO families
          if(rank0_)
            {
              grvy_printf(INFO,"\nCreating HDF5 group for defined IO families\n");
              grvy_printf(INFO,"--> %s : %s\n",fam.group_.c_str(),fam.description_.c_str());
            }

          hid_t group     = -1;
          hid_t dataspace = -1;

          if (rank0_ || (config.RestartSerial() != "write") )
            {
              dataspace = H5Screate_simple(1, dims, NULL);
              assert(dataspace >= 0);
              group = H5Gcreate(file,fam.group_.c_str(),H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
              assert(group >= 0);
            }

          // get pointer to raw data
          double *data = fam.pfunc_->HostReadWrite();
          // special case if writing a serial restart
          if ( (config.RestartSerial() == "write") && (nprocs_ > 1) ){
            serialize_soln_for_write(fam.group_);
            if(rank0_)
              data = serial_soln->GetData();
          }

          // get defined variables for this IO family
          vector<IOVar> vars = ioData.vars_[fam.group_];

          // save raw data
          if (mpi.Root() || (config.RestartSerial() != "write") )
            {
              for(auto var : vars)
                write_soln_data(group,var.varName_,dataspace,&data[var.index_*dims[0]]);
            }

          if(group >= 0)
            H5Gclose(group);
          if(dataspace >= 0 )
            H5Sclose(dataspace);

        }  // end loop over IO families

      if(file >= 0)
        H5Fclose(file);
    }
  else          // read mode
    {

      if(rank0_)
	cout << "Reading in solutiond data from restart..." << endl;

      // verify Dofs match expectations with current mesh
      if (mpi.Root() || (config.RestartSerial() != "read") )
	{
          hid_t dataspace;
	  data_soln = H5Dopen2(file, "/solution/density",H5P_DEFAULT); assert(data_soln >= 0);
	  dataspace = H5Dget_space(data_soln);
	  numInSoln = H5Sget_simple_extent_npoints(dataspace);
	  H5Dclose(data_soln);
	}

      int dof = vfes->GetNDofs();
      if(config.RestartSerial() == "read")
	{
	  int dof_global;
	  MPI_Reduce(&dof, &dof_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  dof = dof_global;
	}

      if( loadFromAuxSol )
        dof = aux_vfes->GetNDofs();

      if (rank0_ || (config.RestartSerial() != "read") )
	assert(numInSoln == dof);

      //-------------------------------------------------------
      // Loop over defined IO families to read desired output
      //-------------------------------------------------------
      for(auto fam : ioData.families_)
        {
          // get pointer to raw data
          double *data = fam.pfunc_->HostReadWrite();
          // special case if starting from aux soln
          if( loadFromAuxSol )
            {
              data = aux_U->HostReadWrite();

              // ks note: would need to add additional logic to handle read of
              // multiple pargridfunctions when changing the solution order
              assert(ioData.families_.size() == 1);
            }

          vector<IOVar> vars = ioData.vars_[fam.group_];
          for(auto var : vars)
            {
              std::string h5Path = fam.group_ + "/" + var.varName_;
              if(rank0_)
                grvy_printf(INFO,"--> Reading h5 path = %s\n",h5Path.c_str());
              if(config.RestartSerial() != "read")
                read_partitioned_soln_data(file,h5Path.c_str(),var.index_*numInSoln,data);
              else
                read_serialized_soln_data (file,h5Path.c_str(),dof,var.index_,data);
            }
        }

      if(file >= 0)
	H5Fclose(file);
    }

  if(mode=="read" && loadFromAuxSol)
    {

      if (mpi.Root())
        cout << "Interpolating from auxOrder = " << auxOrder << " to order = " << order << endl;

      // Interpolate from aux to new order
      U->ProjectGridFunction(*aux_U);

      // If interpolation was successful, the L2 norm of the
      // difference between the auxOrder and order versions should be
      // zero (well... to within round-off-induced error)
      VectorGridFunctionCoefficient lowOrderCoeff(aux_U);
      double err = U->ComputeL2Error(lowOrderCoeff);

      if (mpi.Root())
        cout << "|| interpolation error ||_2 = " << err << endl;

      // Update primitive variables.  This will be done automatically
      // before taking a time step, but we may write paraview output
      // prior to that, in which case the primitives are incorrect.
      // We would like to just call rhsOperator::updatePrimitives, but
      // rhsOperator has not been constructed yet.  As a workaround,
      // that code is duplicated here.
      double *dataUp = Up->HostReadWrite();
      double *x = U->HostReadWrite();
      for(int i=0;i<vfes->GetNDofs();i++)
        {
          Vector iState(num_equation);
          for(int eq=0;eq<num_equation;eq++) iState[eq] = x[i+eq*vfes->GetNDofs()];
          double p = eqState->ComputePressure(iState,dim);
          dataUp[i                   ] = iState[0];
          dataUp[i+  vfes->GetNDofs()] = iState[1]/iState[0];
          dataUp[i+2*vfes->GetNDofs()] = iState[2]/iState[0];
          if(dim==3) dataUp[i+3*vfes->GetNDofs()] = iState[3]/iState[0];
          dataUp[i+(num_equation-1)*vfes->GetNDofs()] = p;
        }

      // clean up aux data
      delete aux_U;
      delete aux_U_data;
      delete aux_vfes;
      delete aux_fec;
    }

#ifdef HAVE_GRVY
  grvy_timer_end(__func__);
#endif
  return;
}

void M2ulPhyS::partitioning_file_hdf5(std::string mode)
{
  grvy_timer_begin(__func__);

  // only rank 0 writes partitioning file
  if( !rank0_ && (mode == "write") )
    return;

  hid_t file, dataspace, data_soln;
  herr_t status;
  std::string fileName = config.GetPartitionBaseName();
  fileName += "." + std::to_string(nprocs_) + "p.h5";

  assert( (mode == "read") || (mode == "write") );

  if(mode == "write")
    {
      assert(partitioning_.Size() > 0);

      if(file_exists(fileName))
	grvy_printf(WARN,"Removing existing partition file: %s\n",fileName.c_str());
      else
	grvy_printf(INFO,"Saving original domain decomposition partition file: %s\n",fileName.c_str());

      file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    }
  else if (mode == "read")
    {
      if( file_exists(fileName))
	{
	  file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	}
      else
	{
	  grvy_printf(ERROR,"[ERROR]: Unable to access necessary partition file on restart -> %s\n",fileName.c_str());
	  exit(ERROR);
	}

      assert(file >= 0);
    }


  if(mode == "write" )
    {
      // Attributes
      h5_save_attribute(file,"numProcs",mpi.WorldSize());

      // Raw partition info
      hsize_t dims[1];
      hid_t data;

      dims[0]   = partitioning_.Size();
      dataspace = H5Screate_simple(1, dims, NULL);
      assert(dataspace >= 0);

      data = H5Dcreate2(file, "partitioning", H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data >= 0);
      status = H5Dwrite(data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, partitioning_.GetData());
      assert(status >= 0);
      H5Dclose(data);
      H5Fclose(file);
    }
  else if (mode == "read")
    {
      partitioning_.SetSize(nelemGlobal_);

      if(rank0_)
	{
	  grvy_printf(INFO,"Reading original domain decomposition partition file: %s\n",fileName.c_str());

	  // verify partition info matches current proc count
	  {
	    int np=0;
	    h5_read_attribute(file,"numProcs",np);
	    grvy_printf(INFO,"--> # partitions defined = %i\n",np);
	    if(np != nprocs_)
	      {
		grvy_printf(ERROR,"[ERROR]: Partition info does not match current processor count -> %i\n",nprocs_);
		exit(ERROR);
	      }
	  }

	  // read partition vector
	  hid_t data;
	  hsize_t numInFile;
	  data = H5Dopen2(file, "partitioning",H5P_DEFAULT); assert(data >= 0);
	  dataspace = H5Dget_space(data);
	  numInFile = H5Sget_simple_extent_npoints(dataspace);

	  assert(numInFile == nelemGlobal_);

	  status = H5Dread(data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, partitioning_.GetData());
	  assert(status >= 0);
	  H5Dclose(data);
	} // <-- end rank0_

#if 0
      // distribute partition vector to all procs
      MPI_Bcast( partitioning_.GetData(),nelemGlobal_, MPI_INT, 0, MPI_COMM_WORLD);
#endif

      // distribute partition vector to all procs (serialzed per process variant)
      int tag = 21;

      if(rank0_)
	{
	  for(int rank=1;rank<nprocs_;rank++)
	    {
	      MPI_Send(partitioning_.GetData(),nelemGlobal_,MPI_INT,rank,tag,MPI_COMM_WORLD);
	      grvy_printf(DEBUG,"Sent partitioning data to rank %i\n",rank);
	    }
	}
      else
	MPI_Recv(partitioning_.GetData(),nelemGlobal_,MPI_INT,0,tag,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if(rank0_)
	grvy_printf(INFO,"--> partition file read complete\n");
    }
  grvy_timer_end(__func__);
}

void M2ulPhyS::serialize_soln_for_write(string ioFamily)
{
  // Get total number of elements
  const int local_ne = mesh->GetNE();
  int global_ne;
  MPI_Reduce(&local_ne, &global_ne, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // ks note: need to double check this routine when additional solution
  // families are added and then remove next assert
  assert(ioFamily == "/solution");

  // get pargridfunction for this IO family
  int iFamily = ioData.getIOFamilyIndex(ioFamily);
  assert(iFamily >= 0);
  ParGridFunction *pfunc = ioData.families_[iFamily].pfunc_;

  if(rank0_)
    assert(global_ne == nelemGlobal_);

  assert( (locToGlobElem != NULL) && (partitioning_ != NULL) );

  if (rank0_)
    {
      grvy_printf(INFO,"Generating serialized restart file...\n");
      // copy my own data
      Array<int> lvdofs, gvdofs;
      Vector lsoln;
      for(int elem=0;elem<local_ne;elem++) {
	int gelem = locToGlobElem[elem];
	vfes->GetElementVDofs(elem,lvdofs);
	pfunc->GetSubVector(lvdofs, lsoln);
	serial_fes->GetElementVDofs(gelem, gvdofs);
	serial_soln->SetSubVector(gvdofs, lsoln);
      }

      // have rank 0 receive data from other tasks and copy its own
      for (int gelem=0; gelem<global_ne; gelem++) {
	int from_rank=partitioning_[gelem];
	if (from_rank!=0) {
	  serial_fes->GetElementVDofs(gelem, gvdofs);
	  lsoln.SetSize(gvdofs.Size());

	  MPI_Recv(lsoln.GetData(), gvdofs.Size(), MPI_DOUBLE,
		   from_rank, gelem, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	  serial_soln->SetSubVector(gvdofs, lsoln);
	}
      }

    }
  else
    {
      // have non-zero ranks send their data to rank 0
      Array<int> lvdofs;
      Vector lsoln;
      for(int elem=0;elem<local_ne;elem++) {
	int gelem = locToGlobElem[elem];
	assert(gelem > 0);
	vfes->GetElementVDofs(elem,lvdofs);
	pfunc->GetSubVector(lvdofs, lsoln); // work for gpu build?

	// send to task 0
	MPI_Send(lsoln.GetData(), lsoln.Size(), MPI_DOUBLE, 0, gelem, MPI_COMM_WORLD);
      }
    }
}  // end function: serialize_soln_for_write()

// convenience function to read solution data for parallel restarts
void M2ulPhyS::read_partitioned_soln_data(hid_t file, string varName, size_t index, double *data)
{
  assert(config.RestartSerial() != "read");

  hid_t data_soln;
  herr_t status;

  data_soln = H5Dopen2(file,varName.c_str(),H5P_DEFAULT);
  assert(data_soln >= 0);
  status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[index]);
  assert(status >= 0);
  H5Dclose(data_soln);
}

// convenience function to read and distribute solution data for serialized restarts
void M2ulPhyS::read_serialized_soln_data(hid_t file, string varName, int numDof, int varOffset,double *data)
{

  assert(config.RestartSerial() == "read");

  hid_t  data_soln;
  herr_t status;
  int    numStateVars;

  const int tag = 20;

  assert( (dim == 2) || (dim == 3) );
  if(dim == 2)
    numStateVars = 4;
  else
    numStateVars = 5;

  if(rank0_)
    {
      grvy_printf(INFO,"[RestartSerial]: Reading %s for distribution\n",varName.c_str());

      Vector data_serial;
      data_serial.SetSize(numDof);
      data_soln = H5Dopen2(file,varName.c_str(),H5P_DEFAULT);
      assert(data_soln >= 0);
      status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,data_serial.GetData());
      assert(status >= 0);
      H5Dclose(data_soln);

      // assign solution owned by rank 0
      assert(partitioning_ != NULL);

      Array<int> lvdofs, gvdofs;
      Vector lnodes;
      int counter = 0;
      int ndof_per_elem;

      for(int gelem=0;gelem<nelemGlobal_;gelem++)
	{
	  if(partitioning_[gelem] == 0)
	    {

	      // cull out subset of local vdof vector to use for this solution var
	      vfes->GetElementVDofs(counter,lvdofs);
	      int numDof_per_this_elem = lvdofs.Size() / numStateVars;
	      int ldof_start_index = varOffset*numDof_per_this_elem;

	      // cull out global vdofs - [subtle note]: we only use the
	      // indexing corresponding to the first state vector reference in
	      // gvdofs() since data_serial() holds the solution for a single state vector
	      serial_fes->GetElementVDofs(gelem, gvdofs);
	      int gdof_start_index = 0;

	      for(int i=0;i<numDof_per_this_elem;i++)
		data[ lvdofs[i+ldof_start_index] ] = data_serial[ gvdofs[i+gdof_start_index] ];
	      counter++;
	    }
	}

      // pack remaining data and send to other processors
      for(int rank=1;rank<nprocs_;rank++)
	{
	  std::vector<double> packedData;
	  for(int gelem=0;gelem<nelemGlobal_;gelem++)
	    if(partitioning_[gelem] == rank)
	      {
		serial_fes->GetElementVDofs(gelem, gvdofs);
		int numDof_per_this_elem = gvdofs.Size() / numStateVars;
		for(int i=0;i<numDof_per_this_elem;i++)
		  packedData.push_back(data_serial[ gvdofs[i] ]);

	      }
	  int tag = 20;
	  MPI_Send(packedData.data(),packedData.size(),MPI_DOUBLE,rank,tag,MPI_COMM_WORLD);
	}

    }  // <-- end rank 0
  else
    {
      int numlDofs = U->Size() / numStateVars;
      grvy_printf(DEBUG,"[%i]: local number of state vars to receive = %i (var=%s)\n",rank_,numlDofs,varName.c_str());

      std::vector<double> packedData(numlDofs);

      // receive solution data from rank 0
      MPI_Recv(packedData.data(),numlDofs,MPI_DOUBLE,0,tag,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // update local state vector
      for(size_t i=0;i<numlDofs;i++)
	data[i + varOffset*numlDofs] = packedData[i];
    }
}

// convenience function to write HDF5 data
void M2ulPhyS::write_soln_data(hid_t group, string varName, hid_t dataspace, double *data)
{
  hid_t data_soln;
  herr_t status;
  assert(group >= 0);

  if(rank0_)
    grvy_printf(INFO,"  --> Saving (%s)\n",varName.c_str());

  data_soln = H5Dcreate2(group,varName.c_str(), H5T_NATIVE_DOUBLE, dataspace,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  assert(data_soln >= 0);

  status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  assert(status >= 0);
  H5Dclose(data_soln);

  return;
}

// ---------------------------------------------
// Routines for I/O data organizer helper class
// ---------------------------------------------

// register a new IO family which maps to a ParGridFunction
void IODataOrganizer::registerIOFamily(std::string description, std::string group, ParGridFunction *pfunc)
{
  IOFamily family{description,group,pfunc};
  std::vector<IOVar> vars;

  families_.push_back(family);
  vars_[group] = vars;

  return;
}

// register individual variables for IO family
void IODataOrganizer::registerIOVar(std::string group, std::string varName, int index)
{
  IOVar newvar{varName,index};
  vars_[group].push_back(newvar);

  return;
}

// return the index to IO family given a group name
int IODataOrganizer::getIOFamilyIndex(std::string group)
{
  for(int i=0;i<families_.size();i++)
    if(families_[i].group_ == group)
      return(i);

  return(-1);
}
