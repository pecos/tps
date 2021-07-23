#include "M2ulPhyS.hpp"
#include <hdf5.h>
#include "utils.hpp"

void M2ulPhyS::restart_files_hdf5(string mode)
{
#ifdef HAVE_GRVY
  grvy_timer_begin(__func__);
#endif

  hid_t file, dataspace, data_soln;
  herr_t status;

  string serialName = "restart_";
  serialName.append( config.GetOutputName() );
  serialName.append( ".sol.h5" );

  string fileName = groupsMPI->getParallelName( serialName );

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
      if (mpi.Root() || !config.SingleRestartFile() )
        {
          file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
          assert(file >= 0);
        }
    }
  else if (mode == "read")
    {

      // verify we have all desired files
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

  // -------------------------------------------------------------------
  // Attributes - current iteration and timing info saved as attributes
  // -------------------------------------------------------------------

  hid_t aid, attr;
  if (mode == "write")
    {
      if (mpi.Root() || !config.SingleRestartFile() )
        {
          aid  = H5Screate(H5S_SCALAR);
          assert(aid >= 0);

          // current iteration count
          attr = H5Acreate(file,"iteration", H5T_NATIVE_INT, aid, H5P_DEFAULT, H5P_DEFAULT);
          assert(attr >= 0);
          status = H5Awrite(attr,H5T_NATIVE_INT,&iter);
          assert(status >= 0);
          H5Aclose(attr);

          // total time
          attr = H5Acreate(file,"time", H5T_NATIVE_DOUBLE, aid, H5P_DEFAULT, H5P_DEFAULT);
          assert(attr >= 0);
          status = H5Awrite(attr,H5T_NATIVE_DOUBLE,&time);
          assert(status >= 0);
          H5Aclose(attr);

          // timestep
          attr = H5Acreate(file,"dt", H5T_NATIVE_DOUBLE, aid, H5P_DEFAULT, H5P_DEFAULT);
          assert(attr >= 0);
          status = H5Awrite(attr,H5T_NATIVE_DOUBLE,&dt);
          assert(status >= 0);  
          H5Aclose(attr);

          // solution order
          attr = H5Acreate(file,"order", H5T_NATIVE_INT, aid, H5P_DEFAULT, H5P_DEFAULT);
          assert(attr >= 0);
          status = H5Awrite(attr,H5T_NATIVE_INT,&order);
          assert(status >= 0);
          H5Aclose(attr);

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
#endif

          H5Sclose(aid);
        }
    }
  else	// read
    {
      attr   = H5Aopen_name(file,"iteration");
      status = H5Aread(attr,H5T_NATIVE_INT,&iter);
      assert(status >= 0);
      H5Aclose(attr);

      attr   = H5Aopen_name(file,"time");
      status = H5Aread(attr,H5T_NATIVE_DOUBLE,&time);
      assert(status >= 0);
      H5Aclose(attr);

      attr   = H5Aopen_name(file,"dt");
      status = H5Aread(attr,H5T_NATIVE_DOUBLE,&dt);
      assert(status >= 0);
      H5Aclose(attr);

      int read_order;
      attr   = H5Aopen_name(file,"order");
      status = H5Aread(attr,H5T_NATIVE_INT,&read_order);
      assert(status >= 0);
      H5Aclose(attr);

      if( loadFromAuxSol )
        {
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

      grvy_printf(GRVY_INFO,"Restarting from iteration = %i\n",iter);
      grvy_printf(GRVY_INFO,"--> time = %e\n",time);
      grvy_printf(GRVY_INFO,"--> dt   = %e\n",dt);

    }

  // -------------------------------------------------------------------
  // Read/write solution state vector
  // -------------------------------------------------------------------

  hsize_t dims[1];
  hsize_t maxdims[1];
  hsize_t numInSoln;

  if(mode == "write")
    {
      if ( config.SingleRestartFile() ){
        assert( (locToGlobElem!=NULL) && (partition!=NULL) );
        dims[0] = serial_fes->GetNDofs();
      } else {
        dims[0] = vfes->GetNDofs();
      }

      hid_t group;
      if (mpi.Root() || !config.SingleRestartFile() )
        {
          // save individual state varbiales from (U) in an HDF5 group named "solution"
          dataspace = H5Screate_simple(1, dims, NULL);
          assert(dataspace >= 0);

          group = H5Gcreate(file,"solution",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
          assert(group >= 0);
        }

      // state vectors in U -> rho, rho-u, rho-v, rho-w, and rho-E
      double *dataU = U->HostReadWrite();

      // if requested single file, serialize
      if ( config.SingleRestartFile() ){
        serialize_soln_for_write();
        dataU = serial_soln->GetData();
      }

      if (mpi.Root() || !config.SingleRestartFile())
        {
          data_soln = H5Dcreate2(group, "density", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU);
          assert(status >= 0);
          H5Dclose(data_soln);

          data_soln = H5Dcreate2(group, "rho-u", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[1*dims[0]]);
          assert(status >= 0);
          H5Dclose(data_soln);

          data_soln = H5Dcreate2(group, "rho-v", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[2*dims[0]]);
          assert(status >= 0);
          H5Dclose(data_soln);

          size_t rhoeIndex = 3*dims[0];

          if(dim == 3)
            {
              rhoeIndex = 4*dims[0];
              data_soln = H5Dcreate2(group, "rho-w", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
              assert(data_soln >= 0);
              status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[3*dims[0]]);
              assert(status >= 0);
              H5Dclose(data_soln);
            }

          data_soln = H5Dcreate2(group, "rho-E", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[rhoeIndex]);
          assert(status >= 0);
          H5Dclose(data_soln);

#ifdef SAVE_PRIMITIVE_VARS

          // not set up to save primitives to a single restart, so punt
          assert(!config.SingleRestartFile());

          // update primitive to latest timestep prior to write
          rhsOperator->updatePrimitives(*U);

          data_soln = H5Dcreate2(group, "u-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[1*dims[0]]);
          assert(status >= 0);
          H5Dclose(data_soln);

          data_soln = H5Dcreate2(group, "v-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[2*dims[0]]);
          assert(status >= 0);
          H5Dclose(data_soln);

          size_t pIndex = 3*dims[0];

          if(dim == 3)
            {
              pIndex = 4*dims[0];

              data_soln = H5Dcreate2(group, "w-velocity", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
              assert(data_soln >= 0);
              status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[3*dims[0]]);
              assert(status >= 0);
              H5Dclose(data_soln);
            }

          data_soln = H5Dcreate2(group, "pressure", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          assert(data_soln >= 0);
          status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataUp[pIndex]);
          assert(status >= 0);
          H5Dclose(data_soln);
#endif

          H5Sclose(dataspace);
          H5Gclose(group);
          H5Fclose(file);
        }
    }
  else          // read mode
    {

      cout << "Reading in state vector from restart..." << endl;
      double *dataU;
      if( loadFromAuxSol )
	  dataU = aux_U->HostReadWrite();
      else
	  dataU = U->HostReadWrite();

      data_soln = H5Dopen2(file, "/solution/density",H5P_DEFAULT); assert(data_soln >= 0);
      dataspace = H5Dget_space(data_soln);
      numInSoln = H5Sget_simple_extent_npoints(dataspace);

      // verify Dofs match expectations
      int dof = vfes->GetNDofs();
      if( loadFromAuxSol )
        dof = aux_vfes->GetNDofs();

      assert(numInSoln == dof);

      status    = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataU);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dopen2(file, "/solution/rho-u",H5P_DEFAULT); assert(data_soln >= 0);
      status    = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[1*numInSoln]);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dopen2(file, "/solution/rho-v",H5P_DEFAULT); assert(data_soln >= 0);
      status    = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[2*numInSoln]);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dopen2(file, "/solution/rho-w",H5P_DEFAULT); assert(data_soln >= 0);
      status    = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[3*numInSoln]);
      assert(status >= 0);
      H5Dclose(data_soln);

      data_soln = H5Dopen2(file, "/solution/rho-E",H5P_DEFAULT); assert(data_soln >= 0);
      status    = H5Dread(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dataU[4*numInSoln]);
      assert(status >= 0);
      H5Dclose(data_soln);

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


void M2ulPhyS::serialize_soln_for_write()
{
  // Get total number of elements
  const int local_ne = mesh->GetNE();
  int global_ne;
  MPI_Reduce(&local_ne, &global_ne, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  const int rank = mpi.WorldRank();

  assert( (locToGlobElem!=NULL) && (partition!=NULL) );

  if (rank==0) {
    // copy my own data
    Array<int> lvdofs, gvdofs;
    Vector lsoln;
    for(int elem=0;elem<local_ne;elem++) {
      int gelem = locToGlobElem[elem];
      vfes->GetElementVDofs(elem,lvdofs);
      U->GetSubVector(lvdofs, lsoln);
      serial_fes->GetElementVDofs(gelem, gvdofs);
      serial_soln->SetSubVector(gvdofs, lsoln);
    }


    // have rank 0 receive data from other tasks and copy its own
    for (int gelem=0; gelem<global_ne; gelem++) {
      int from_rank=partition[gelem];
      if (from_rank!=0) {

        serial_fes->GetElementVDofs(gelem, gvdofs);
        lsoln.SetSize(gvdofs.Size());

        MPI_Recv(lsoln.GetData(), gvdofs.Size(), MPI_DOUBLE,
                 from_rank, gelem, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        serial_soln->SetSubVector(gvdofs, lsoln);
      }
    }

  } else {
    // have non-zero ranks send their data to rank 0
    Array<int> lvdofs;
    Vector lsoln;
    for(int elem=0;elem<local_ne;elem++) {
      int gelem = locToGlobElem[elem];
      vfes->GetElementVDofs(elem,lvdofs);
      U->GetSubVector(lvdofs, lsoln); // work for gpu build?

      // send to task 0
      MPI_Send(lsoln.GetData(), lsoln.Size(), MPI_DOUBLE, 0, gelem, MPI_COMM_WORLD);
    }
  }

}

