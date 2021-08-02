#include "io.hpp"

IO_operations::IO_operations(string filePrefix,
                             MPI_Groups *_groupMPI,
                             RunConfiguration *_config,
                             int* _locToGlobElem,
                             Array<int> &_partitioning_,
                             const int &_defaultPartMethod,
                             int &_num_equation,
                             double &_time,
                             double &_dt,
                             int &_iter,
                             int &_order,
                             int &_dim
                            ):
groupsMPI(_groupMPI),
config(_config),
locToGlobElem(_locToGlobElem),
partitioning_(_partitioning_),
defaultPartMethod(_defaultPartMethod),
num_equation(_num_equation),
time(_time),
dt(_dt),
iter(_iter),
order(_order),
dim(_dim)
{
  prefix_serial = filePrefix;
  
  nprocs_ = groupsMPI->getSession()->WorldSize();
  rank_   = groupsMPI->getSession()->WorldRank();
  rank0_  = groupsMPI->getSession()->Root();
  
  loadFromAuxSol = config->RestartFromAux();

  serial_mesh = NULL;
  serial_fes  = NULL;
  serial_soln = NULL;
  
  serial_rmsFes = NULL;
  serial_rms = NULL;
  
  U      = NULL;
  meanUp = NULL;
  rms    = NULL;
  rmsFes = NULL;
}

IO_operations::~IO_operations()
{
  if ( rank0_ && (config->RestartSerial() != "no") )
  {
    delete serial_mesh;
    delete serial_fes;
    delete serial_soln;
    if( meanUp!=NULL )
    {
      delete serial_rmsFes;
      delete serial_rms;
    }
  }
}

void IO_operations::setSolutionGridFunction(ParGridFunction* _U)
{
  U = _U;
  
  vfes = U->ParFESpace();
  mesh = vfes->GetParMesh();
  fec  = vfes->FEColl();
  
  if ( rank0_ && (config->RestartSerial() != "no") )
  {
    serial_mesh = new Mesh(config->GetMeshFileName().c_str() );
    serial_fes = new FiniteElementSpace(serial_mesh, fec, num_equation, Ordering::byNODES);
    serial_soln = new GridFunction(serial_fes);
    // to help detect errors, initialize to nan
    *serial_soln = std::numeric_limits<double>::quiet_NaN();
  }
  
  // compute total number of elements
  int locElems = vfes->GetNE();
  MPI_Allreduce(&locElems, &nelemGlobal_,
                1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}


void IO_operations::setAveragesObject(Averaging* _averages)
{
  average = _averages;
  
  if( average->ComputeMean() )
  {
    meanUp = average->GetMeanUp();
    rms = average->GetRMS();
    rmsFes = average->GetRMSFes();
    
    if ( rank0_ && (config->RestartSerial() != "no") )
    {
      serial_meanSol = new GridFunction(serial_fes);
      serial_rmsFes = new FiniteElementSpace(serial_mesh, fec, 6, Ordering::byNODES);
      serial_rms    = new GridFunction(serial_rmsFes);
    }
  }
}


void IO_operations::restart_files_hdf5(string mode)
{
  if( U==NULL )
  {
    cout<<"IO Error: solution not set"<<endl;
    return;
  }
  if( (meanUp==NULL || rms==NULL) && average->ComputeMean() )
  {
    cout<<"IO Error: averages not set"<<endl;
    return;
  }
#ifdef HAVE_GRVY
  grvy_timer_begin(__func__);
#endif

  hid_t file = NULL;
  hid_t dataspace,data_soln;
  herr_t status;
  Vector dataSerial;

  //string serialName = "restart_";
  string serialName = prefix_serial;
  serialName.append( config->GetOutputName() );
  serialName.append( ".sol.h5" );
  string fileName;

  if( ((config->RestartSerial() == "read") && (mode == "read")) || 
       (config->RestartSerial() == "write") )
    fileName  = serialName;
  else
    fileName = groupsMPI->getParallelName( serialName );

  // Variables used if (and only if) restarting from different order
  FiniteElementCollection *aux_fec    = NULL;
  ParFiniteElementSpace   *aux_vfes   = NULL;
  ParGridFunction         *aux_U      = NULL;
  double                  *aux_U_data = NULL;
  int auxOrder = -1;

  if(groupsMPI->getSession()->Root())
    cout << "HDF5 restart files mode: " << mode << endl;

  assert( (mode == "read") || (mode == "write") );

  // open restart files (currently per MPI process variants)
  if (mode == "write")
  {
    if (rank0_ || (config->RestartSerial() != "write") )
    {
      file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
      assert(file >= 0);
    }
  }
  else if (mode == "read")
  {

    if(config->RestartSerial() == "read")
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
    if (rank0_ || (config->RestartSerial() != "write") )
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
      if( average->ComputeMean() )
      {
        // samples meanUp
        h5_save_attribute(file,"samplesMean",average->GetSamplesMean());
        h5_save_attribute(file,"samplesInterval",average->GetSamplesInterval());
      }
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

    // normal restarts have each process read their own portion of the
    // solution; a serial restart only reads on rank 0 and distributes to
    // the remaining processes

    if (rank0_ || (config->RestartSerial() != "read") )
    {
      h5_read_attribute(file,"iteration",iter);
      h5_read_attribute(file,"time",time);
      h5_read_attribute(file,"dt",dt);
      h5_read_attribute(file,"order",read_order);
      if( average->ComputeMean())
      {
        int samplesMean, intervals;
        h5_read_attribute(file,"samplesMean",samplesMean);
        h5_read_attribute(file,"samplesInterval",intervals);
        average->SetSamplesMean(samplesMean);
        average->SetSamplesInterval(intervals);
      }
    }

    if(config->RestartSerial() == "read")
    {
      MPI_Bcast(&iter,      1, MPI_INT,    0, MPI_COMM_WORLD);
      MPI_Bcast(&time,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&dt,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&read_order,1, MPI_INT,    0, MPI_COMM_WORLD);
      if( average->ComputeMean())
      {
        int sampMean = average->GetSamplesMean();
        int intervals = average->GetSamplesInterval();
        MPI_Bcast(&sampMean,    1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&intervals   ,1, MPI_INT, 0, MPI_COMM_WORLD);
      }
    }

    if( loadFromAuxSol )
    {
      assert(config->RestartSerial() == "no");
      auxOrder = read_order;

      if( config->GetBasisType() == 0 )
        aux_fec = new DG_FECollection(auxOrder, dim, BasisType::GaussLegendre);
      else if ( config->GetBasisType() == 1 )
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
      if( average->ComputeMean() )
      {
        grvy_printf(GRVY_INFO,"Restarting averages with %i\n samples",
                    average->GetSamplesMean());
      }
    }

  }

  // -------------------------------------------------------------------
  // Read/write solution state vector
  // -------------------------------------------------------------------

  hsize_t dims[1];
  hsize_t maxdims[1];
  hsize_t numInSoln;

  if(mode == "write")
  {
    if ( (config->RestartSerial() == "write") && (nprocs_ > 1) )
    {
      assert( (locToGlobElem != NULL) && (partitioning_ != NULL) );
      if( rank0_ ) dims[0] = serial_fes->GetNDofs();
    }
    else
      dims[0] = vfes->GetNDofs();

    hid_t group, groupMeanUp, groupRMS;
    if ( rank0_ || (config->RestartSerial() != "write") )
    {
      // save individual state varbiales from (U) in an HDF5 group named "solution"
      dataspace = H5Screate_simple(1, dims, NULL);
      assert(dataspace >= 0);

      group = H5Gcreate(file,"solution",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      assert(group >= 0);
      if( average->ComputeMean() )
      {
        groupMeanUp = H5Gcreate(file,"meanSolution",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        groupRMS    = H5Gcreate(file,"rmsData"     ,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        assert(groupMeanUp >= 0);
        assert(groupRMS    >= 0);
      }
    }

    // state vectors in U -> rho, rho-u, rho-v, rho-w, and rho-E
    double *dataU = U->HostReadWrite();
    double *dataMeanUp = NULL;
    double *dataRMS = NULL;

    if( average->ComputeMean())
    {
      dataMeanUp = meanUp->HostReadWrite();
      dataRMS    = rms->HostReadWrite();
    }

    // if requested single file restart, serialize
    if ( (config->RestartSerial() == "write") && (nprocs_ > 1) )
    {
      serialize_soln_for_write(false);
      if( rank0_ ) dataU = serial_soln->HostReadWrite();
      if( average->ComputeMean() )
      {
        serialize_soln_for_write(true);
        if( rank0_ ) dataMeanUp = serial_meanSol->HostReadWrite();
        if( rank0_ ) dataRMS    = serial_rms->HostReadWrite();
      }
    }

    if (rank0_ || (config->RestartSerial() != "write") )
    {
      data_soln = H5Dcreate2(group, "density", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data_soln >= 0);
      status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dcreate2(group, "rho-u", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data_soln >= 0);
      status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU+dims[0]);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dcreate2(group, "rho-v", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data_soln >= 0);
      status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU+2*dims[0]);
      assert(status >= 0);
      H5Dclose(data_soln);

      size_t rhoeIndex = 3*dims[0];

      if(dim == 3)
      {
        rhoeIndex = 4*dims[0];
        data_soln = H5Dcreate2(group, "rho-w", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU+3*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);
      }

      data_soln = H5Dcreate2(group, "rho-E", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data_soln >= 0);
      status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU+rhoeIndex);
      assert(status >= 0);
      H5Dclose(data_soln);
      
      if( average->ComputeMean() )
      {
        data_soln = H5Dcreate2(groupMeanUp, "meanDens", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataMeanUp);
        assert(status >= 0);
        H5Dclose(data_soln);

        data_soln = H5Dcreate2(groupMeanUp, "mean-u", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataMeanUp+dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);

        data_soln = H5Dcreate2(groupMeanUp, "mean-v", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataMeanUp+2*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);

        size_t pIndex = 3*dims[0];

        if(dim == 3)
        {
          pIndex = 4*dims[0];
          data_soln = H5Dcreate2(groupMeanUp, "mean-w", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataMeanUp+3*dims[0]);
          assert(status >= 0);
          H5Dclose(data_soln);
        }

        data_soln = H5Dcreate2(groupMeanUp, "mean-p", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataMeanUp+pIndex);
        assert(status >= 0);
        H5Dclose(data_soln);
        
        // RMS DATA
        data_soln = H5Dcreate2(groupRMS, "uu", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS);
        assert(status >= 0);
        H5Dclose(data_soln);

        data_soln = H5Dcreate2(groupRMS, "vv", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS+dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);

        data_soln = H5Dcreate2(groupRMS, "ww", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS+2*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);

        data_soln = H5Dcreate2(groupRMS, "uv", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS+3*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);
        
        data_soln = H5Dcreate2(groupRMS, "uw", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS+4*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);
        
        data_soln = H5Dcreate2(groupRMS, "vw", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(data_soln >= 0);
        status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataRMS+5*dims[0]);
        assert(status >= 0);
        H5Dclose(data_soln);
      }

#ifdef SAVE_PRIMITIVE_VARS

      // not set up to save primitives to a single restart, so punt
      assert(config->RestartSerial() == "no");

      // update primitive to latest timestep prior to write
//       rhsOperator->updatePrimitives(*U);
// 
//       data_soln = H5Dcreate2(group, "u-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//       assert(data_soln >= 0);
//       status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[1*dims[0]]);
//       assert(status >= 0);
//       H5Dclose(data_soln);
// 
//       data_soln = H5Dcreate2(group, "v-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//       assert(data_soln >= 0);
//       status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[2*dims[0]]);
//       assert(status >= 0);
//       H5Dclose(data_soln);
// 
//       size_t pIndex = 3*dims[0];
// 
//       if(dim == 3)
//       {
//         pIndex = 4*dims[0];
// 
//         data_soln = H5Dcreate2(group, "w-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//         assert(data_soln >= 0);
//         status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[3*dims[0]]);
//         assert(status >= 0);
//         H5Dclose(data_soln);
//       }
// 
//       data_soln = H5Dcreate2(group, "pressure", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//       assert(data_soln >= 0);
//       status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[pIndex]);
//       assert(status >= 0);
//       H5Dclose(data_soln);
#endif

      H5Sclose(dataspace);
      H5Gclose(group);
      if( average->ComputeMean() )
      {
        H5Gclose(groupMeanUp);
        H5Gclose(groupRMS);
      }
      H5Fclose(file);
    }
  }
  else          // read mode
  {

    if(rank0_)
      cout << "Reading in state vector from restart..." << endl;

    double *dataU, *dataMeanUp, *dataRMS;
    if( loadFromAuxSol )
      dataU = aux_U->HostReadWrite();
    else
      dataU = U->HostReadWrite();
    
    if( average->ComputeMean())
    {
      dataMeanUp = meanUp->HostReadWrite();
      dataRMS = rms->HostReadWrite();
    }

    // verify Dofs match expectations with current mesh
    if (rank0_ || (config->RestartSerial() != "read") )
    {
      data_soln = H5Dopen2(file, "/solution/density",H5P_DEFAULT);
      assert(data_soln >= 0);
      dataspace = H5Dget_space(data_soln);
      numInSoln = H5Sget_simple_extent_npoints(dataspace);
      H5Dclose(data_soln);
    }

    int dof = vfes->GetNDofs();
    if(config->RestartSerial() == "read")
    {
      int dof_global;
      MPI_Reduce(&dof, &dof_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
      dof = dof_global;
    }

    if( loadFromAuxSol )
      dof = aux_vfes->GetNDofs();

    if (rank0_ || (config->RestartSerial() != "read") )
      assert(numInSoln == dof);

    if(config->RestartSerial() != "read")
    {
      read_partitioned_soln_data(file,"/solution/density",0,          dataU);
      read_partitioned_soln_data(file,"/solution/rho-u",(1*numInSoln),dataU);
      read_partitioned_soln_data(file,"/solution/rho-v",(2*numInSoln),dataU);
      if(dim == 3)
      {
        read_partitioned_soln_data(file,"/solution/rho-w",(3*numInSoln),dataU);
        read_partitioned_soln_data(file,"/solution/rho-E",(4*numInSoln),dataU);
      }
      else
        read_partitioned_soln_data(file,"/solution/rho-E",(3*numInSoln),dataU);
      if( average->ComputeMean() )
      {
        read_partitioned_soln_data(file,"/meanSolution/meanDens",0,          dataMeanUp);
        read_partitioned_soln_data(file,"/meanSolution/mean-u",(1*numInSoln),dataMeanUp);
        read_partitioned_soln_data(file,"/meanSolution/mean-v",(2*numInSoln),dataMeanUp);
        if(dim == 3)
        {
          read_partitioned_soln_data(file,"/meanSolution/mean-w",(3*numInSoln),dataMeanUp);
          read_partitioned_soln_data(file,"/meanSolution/mean-p",(4*numInSoln),dataMeanUp);
        }
        else
          read_partitioned_soln_data(file,"/meanSolution/mean-p",(3*numInSoln),dataMeanUp);
        
        read_partitioned_soln_data(file,"/rmsData/uu",0,            dataRMS);
        read_partitioned_soln_data(file,"/rmsData/vv",(1*numInSoln),dataRMS);
        read_partitioned_soln_data(file,"/rmsData/ww",(2*numInSoln),dataRMS);
        read_partitioned_soln_data(file,"/rmsData/uv",(3*numInSoln),dataRMS);
        read_partitioned_soln_data(file,"/rmsData/uw",(4*numInSoln),dataRMS);
        read_partitioned_soln_data(file,"/rmsData/vw",(5*numInSoln),dataRMS);
      }
    }
    else
    {
      read_serialized_soln_data(file,"/solution/density",dof,0,dataU,false);
      read_serialized_soln_data(file,"/solution/rho-u",  dof,1,dataU,false);
      read_serialized_soln_data(file,"/solution/rho-v",  dof,2,dataU,false);
      if(dim == 3)
      {
        read_serialized_soln_data(file,"/solution/rho-w",  dof,3,dataU,false);
        read_serialized_soln_data(file,"/solution/rho-E",  dof,4,dataU,false);
      }
      else
        read_serialized_soln_data(file,"/solution/rho-E",  dof,3,dataU,false);
      
      if( average->ComputeMean() )
      {
        read_serialized_soln_data(file,"/meanSolution/meanDens",dof,0,dataMeanUp,false);
        read_serialized_soln_data(file,"/meanSolution/mean-u",  dof,1,dataMeanUp,false);
        read_serialized_soln_data(file,"/meanSolution/mean-v",  dof,2,dataMeanUp,false);
        if(dim == 3)
        {
          read_serialized_soln_data(file,"/meanSolution/mean-w",dof,3,dataMeanUp,false);
          read_serialized_soln_data(file,"/meanSolution/mean-p",dof,4,dataMeanUp,false);
        }
        else
          read_serialized_soln_data(file,"/meanSolution/mean-p",dof,3,dataMeanUp,false);
        
        read_serialized_soln_data(file,"/rmsData/uu",dof,0,dataRMS,true);
        read_serialized_soln_data(file,"/rmsData/vv",dof,1,dataRMS,true);
        read_serialized_soln_data(file,"/rmsData/ww",dof,2,dataRMS,true);
        read_serialized_soln_data(file,"/rmsData/uv",dof,3,dataRMS,true);
        read_serialized_soln_data(file,"/rmsData/uw",dof,4,dataRMS,true);
        read_serialized_soln_data(file,"/rmsData/vw",dof,5,dataRMS,true);
      }
    }

    if(file != NULL)
      H5Fclose(file);
  }

  if(mode=="read" && loadFromAuxSol)
  {

    if (rank0_)
      cout << "Interpolating from auxOrder = " << auxOrder << " to order = " << order << endl;

    // Interpolate from aux to new order
    U->ProjectGridFunction(*aux_U);

    // If interpolation was successful, the L2 norm of the
    // difference between the auxOrder and order versions should be
    // zero (well... to within round-off-induced error)
    VectorGridFunctionCoefficient lowOrderCoeff(aux_U);
    double err = U->ComputeL2Error(lowOrderCoeff);

    if (rank0_)
      cout << "|| interpolation error ||_2 = " << err << endl;

    // Update primitive variables.  This will be done automatically
    // before taking a time step, but we may write paraview output
    // prior to that, in which case the primitives are incorrect.
    // We would like to just call rhsOperator::updatePrimitives, but
    // rhsOperator has not been constructed yet.  As a workaround,
    // that code is duplicated here.
//     double *dataUp = Up->HostReadWrite();
//     double *x = U->HostReadWrite();
//     for(int i=0; i<vfes->GetNDofs(); i++)
//     {
//       Vector iState(num_equation);
//       for(int eq=0; eq<num_equation; eq++) iState[eq] = x[i+eq*vfes->GetNDofs()];
//       double p = eqState->ComputePressure(iState,dim);
//       dataUp[i                   ] = iState[0];
//       dataUp[i+  vfes->GetNDofs()] = iState[1]/iState[0];
//       dataUp[i+2*vfes->GetNDofs()] = iState[2]/iState[0];
//       if(dim==3) dataUp[i+3*vfes->GetNDofs()] = iState[3]/iState[0];
//       dataUp[i+(num_equation-1)*vfes->GetNDofs()] = p;
//    }

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


void IO_operations::serialize_soln_for_write(bool averages)
{
  // Get total number of elements
  const int local_ne = mesh->GetNE();
  int global_ne;
  MPI_Reduce(&local_ne, &global_ne, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank0_)
    assert(global_ne == nelemGlobal_);

  assert( (locToGlobElem != NULL) && (partitioning_ != NULL) );

  if (rank0_)
  {
    if( averages ) grvy_printf(INFO,"Generating averages serialized restart file...\n");
    else grvy_printf(INFO,"Generating serialized restart file...\n");
    // copy my own data
    Array<int> lvdofs, gvdofs,lvdofsRMS, gvdofsRMS;
    Vector lsoln, lrms; 
    lsoln.UseDevice(false); // make sure all data movement is happening on CPU
    lrms.UseDevice(false);
    for(int elem=0; elem<local_ne; elem++)
    {
      int gelem = locToGlobElem[elem];
      if( averages && average->ComputeMean())
      {
        vfes->GetElementVDofs(elem,lvdofs);
        meanUp->GetSubVector(lvdofs, lsoln);
        serial_fes->GetElementVDofs(gelem, gvdofs);
        serial_meanSol->SetSubVector(gvdofs, lsoln);
        
        rmsFes->GetElementVDofs(elem,lvdofsRMS);
        rms->GetSubVector(lvdofsRMS, lrms);
        serial_rmsFes->GetElementVDofs(gelem, gvdofsRMS);
        serial_rms->SetSubVector(gvdofsRMS, lrms);
      }else
      {
        vfes->GetElementVDofs(elem,lvdofs);
        U->GetSubVector(lvdofs, lsoln);
        serial_fes->GetElementVDofs(gelem, gvdofs);
        serial_soln->SetSubVector(gvdofs, lsoln);
      }
    }

    Vector tempData; tempData.UseDevice(false);
    Array<int> tempDofs;
    // have rank 0 receive data from other tasks and copy its own
    for (int gelem=0; gelem<global_ne; gelem++)
    {
      int from_rank = partitioning_[gelem];
      if (from_rank!=0)
      {
        if( averages && average->ComputeMean() )
        {
          serial_fes->GetElementVDofs(gelem, gvdofs);
          lsoln.SetSize(gvdofs.Size());
          serial_rmsFes->GetElementVDofs(gelem, gvdofsRMS);
          tempData.SetSize(gvdofs.Size()+gvdofsRMS.Size());
        }else
        {
          serial_fes->GetElementVDofs(gelem, tempDofs);
          tempData.SetSize(tempDofs.Size());
        }

        MPI_Recv(tempData.GetData(), tempData.Size(), MPI_DOUBLE,
                 from_rank, gelem, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if( averages && average->ComputeMean() )
        {
          serial_meanSol->SetSubVector(gvdofs, tempData);
          serial_rms->SetSubVector(gvdofsRMS, 
                                   tempData.GetData()+gvdofs.Size());
        }else
        {
          serial_soln->SetSubVector(tempDofs, tempData);
        }
        
      }
    }

  }
  else
  {
    // have non-zero ranks send their data to rank 0
    Array<int> lvdofs, vdofsRMS;
    Vector lsoln, tempRMS;
    lsoln.UseDevice(false);
    tempRMS.UseDevice(false);
    for(int elem=0; elem<local_ne; elem++)
    {
      int gelem = locToGlobElem[elem];
//       assert(gelem > 0);
      if( averages && average->ComputeMean() )
      {
        vfes->GetElementVDofs(elem,lvdofs);
        rmsFes->GetElementVDofs(elem,vdofsRMS);
        meanUp->GetSubVector(lvdofs,lsoln);
        rms->GetSubVector(vdofsRMS,tempRMS);
        
        Vector temp(lsoln);
        lsoln.SetSize(lvdofs.Size()+vdofsRMS.Size());
        for(int n=0;n<lvdofs.Size();n++) lsoln(n) = temp(n);
        for(int n=0;n<vdofsRMS.Size();n++) lsoln(n+lvdofs.Size()) = tempRMS(n);
      }else
      {
        vfes->GetElementVDofs(elem,lvdofs);
        U->GetSubVector(lvdofs, lsoln); // work for gpu build?
      }

      // send to task 0
      MPI_Send(lsoln.GetData(), lsoln.Size(), MPI_DOUBLE, 0, gelem, MPI_COMM_WORLD);
    }
  }
}

// convenience function to read solution data for parallel restarts
void IO_operations::read_partitioned_soln_data(hid_t file, string varName, size_t index, double *data)
{
  assert(config->RestartSerial() != "read");

  hid_t data_soln;
  herr_t status;

  data_soln = H5Dopen2(file,varName.c_str(),H5P_DEFAULT);
  assert(data_soln >= 0);
  status = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[index]);
  assert(status >= 0);
  H5Dclose(data_soln);
}

// convenience function to read and distribute solution data for serialized restarts
void IO_operations::read_serialized_soln_data(hid_t file, string varName, int numDof, int varOffset,double *data, bool isRMS)
{

  assert(config->RestartSerial() == "read");

  hid_t  data_soln;
  herr_t status;
  int    numStateVars;

  const int tag = 20;

  assert( (dim == 2) || (dim == 3) );
  if(dim == 2)
    numStateVars = 4;
  else
    numStateVars = 5;
  if( isRMS ) numStateVars = 6;

  if(rank0_)
  {
    grvy_printf(INFO,"[RestartSerial]: Reading %s for distribution\n",varName.c_str());

    Vector data_serial;
    data_serial.UseDevice(false);
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
    lnodes.UseDevice(false);
    int counter = 0;
    int ndof_per_elem;

    for(int gelem=0; gelem<nelemGlobal_; gelem++)
    {
      if(partitioning_[gelem] == 0)
      {

        // cull out subset of local vdof vector to use for this solution var
        if (isRMS ) rmsFes->GetElementVDofs(counter,lvdofs);
        else vfes->GetElementVDofs(counter,lvdofs);
        
        int numDof_per_this_elem = lvdofs.Size() / numStateVars;
        int ldof_start_index = varOffset*numDof_per_this_elem;

        // cull out global vdofs - [subtle note]: we only use the
        // indexing corresponding to the first state vector reference in
        // gvdofs() since data_serial() holds the solution for a single state vector
        if( isRMS ) serial_rmsFes->GetElementVDofs(gelem, gvdofs);
        else serial_fes->GetElementVDofs(gelem, gvdofs);
        int gdof_start_index = 0;

        for(int i=0; i<numDof_per_this_elem; i++)
          data[ lvdofs[i+ldof_start_index] ] = data_serial[ gvdofs[i+gdof_start_index] ];
        counter++;
      }
    }

    // pack remaining data and send to other processors
    for(int rank=1; rank<nprocs_; rank++)
    {
      std::vector<double> packedData;
      for(int gelem=0; gelem<nelemGlobal_; gelem++)
        if(partitioning_[gelem] == rank)
        {
          if( isRMS ) serial_rmsFes->GetElementVDofs(gelem, gvdofs);
          else serial_fes->GetElementVDofs(gelem, gvdofs);
          int numDof_per_this_elem = gvdofs.Size() / numStateVars;
          for(int i=0; i<numDof_per_this_elem; i++)
            packedData.push_back(data_serial[ gvdofs[i] ]);

        }
      int tag = 20;
      MPI_Send(packedData.data(),packedData.size(),MPI_DOUBLE,rank,tag,MPI_COMM_WORLD);
    }

  }  // <-- end rank 0
  else
  {
    int numlDofs = U->Size() / numStateVars;
    if( isRMS ) numlDofs = rms->Size()/ numStateVars;
    grvy_printf(DEBUG,"[%i]: local number of state vars to receive = %i (var=%s)\n",rank_,numlDofs,varName.c_str());

    std::vector<double> packedData(numlDofs);

    // receive solution data from rank 0
    MPI_Recv(packedData.data(),numlDofs,MPI_DOUBLE,0,tag,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // update local state vector
    for(size_t i=0; i<numlDofs; i++)
      data[i + varOffset*numlDofs] = packedData[i];
  }
}
