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

#include "radiation.hpp"

#include "utils.hpp"

MFEM_HOST_DEVICE Radiation::Radiation(const RadiationInput &_inputs)
    : inputs(_inputs) {}

MFEM_HOST_DEVICE NetEmission::NetEmission(const RadiationInput &_inputs)
    : Radiation(_inputs), necTable_(LinearTable(inputs.necTableInput)) {
  assert(_inputs.model == NET_EMISSION);
  assert(_inputs.necModel == TABULATED_NEC);
  assert(_inputs.necTableInput.order == 1);
  // To generalize beyond LinearTable, need to have necTable_ be a
  // pointer to TableInterpolator and then instantiate the appropriate
  // class here (e.g., necTable_ = new LinearTable(inputs.necTableInput);).
  // However, this design causes unexplained problems for the CUDA
  // version.  See Issue #184.  To avoid this issue, we use the design
  // here (i.e., necTable_ has type LinearTable).  Since this is the
  // only option currently supported, this approach is ok.  It will
  // have to be revisited if other options are added in the future.
}

MFEM_HOST_DEVICE NetEmission::~NetEmission() {}

P1Model::P1Model(const RadiationInput &_inputs, ParMesh *_mesh, 
                 FiniteElementCollection *_fec, ParFiniteElementSpace *_fes, 
                 bool _axisymmetric)
    :Radiation(_inputs),
     mesh(_mesh),
     fec(_fec),
     fes(_fes),
     axisymmetric(_axisymmetric) {

  assert(inputs.model == P1_MODEL);
  assert(inputs.P1TableInput.order == 1);

  NumOfGroups = inputs.NumOfGroups;
  visualization = inputs.visualization;

  nprocs_ = Mpi::WorldSize();
  rank_ = Mpi::WorldRank();
  rank0_ = Mpi::Root();

  dim = mesh->Dimension();
  order = 1;

  // For Boundary Conditions
  WallEmissivity = 0.2; // It seems that the results are quite insensitive to this parameter.
  rbc_a_val = WallEmissivity / (2.0 * (2.0 - WallEmissivity)) ; // du/dn + a * u = b

  // Set some solver parameters
  MaxIters = inputs.MaxIters;
  P1_Tolerance = inputs.P1_Tolerance;
  PrintLevels = inputs.PrintLevels;
  solve_rad_every_n_ = inputs.solve_rad_every_n;

  // Define a finite element space on the mesh. Here we use H1 continuous
  // high-order Lagrange finite elements of the given order.
  fec_rad = new H1_FECollection(order, dim);
  fes_rad = new ParFiniteElementSpace(mesh, fec_rad);
  dfes_rad = new ParFiniteElementSpace(mesh, fec_rad, dim, Ordering::byNODES);

  dof = fes_rad->GetNDofs();
  total_num_dofs = fes_rad->GlobalTrueVSize();
  fes_rad->GetBoundaryTrueDofs(boundary_dofs);   // Extract the list of all the boundary DOFs. 

  coordsDof = new ParGridFunction(dfes_rad);
  mesh->GetNodes(*coordsDof);

  Radius = new ParGridFunction(fes_rad), *Radius = 0.0;
  OneOverRadius = new ParGridFunction(fes_rad), *OneOverRadius = 0.0;

  if (axisymmetric){ 
    for (int i = 0; i < dof; i++) {
      Vector xyz(3);
      for (int d = 0; d < dim; d++) xyz[d] = (*coordsDof)[i + d * dof]; 
      (*Radius)[i] = xyz[0]; 
      (*OneOverRadius)[i] = 1.0/(xyz[0]+small); 

      // (*Radius)[i] = xyz[1]; 
      // (*OneOverRadius)[i] = 1.0/(xyz[1]+small); 
    }
  }

  if (rank0_){
    std::cout << "Radiation solver parameters: "  <<std::endl;
    std::cout << "--Number of MPI processes = " << nprocs_ <<std::endl;
    std::cout << "--Grid dimensions = " << dim <<std::endl;
    std::cout << "--Number of unknowns: " << total_num_dofs << std::endl;
    std::cout << "--Number of dofs: " << dof << std::endl;
    std::cout << "--Number of Grid Attributes: " << mesh->bdr_attributes.Max() << std::endl;
    std::cout << "--Number of Boundary dofs: " << boundary_dofs.Size() << std::endl;
  }


  Temperature = new ParGridFunction(fes_rad), *Temperature = 0.0;
  energySink_P1 = new ParGridFunction(fes_rad), *energySink_P1 = 0.0;
  energySinkOptIntermediate = new ParGridFunction(fes_rad), *energySinkOptIntermediate = 0.0;

  //    Define the solution block U for all groups. Set
  //    the initial guess to zero, which also sets the boundary conditions.
  vfes_rad = new ParFiniteElementSpace(mesh, fec_rad, NumOfGroups, Ordering::byNODES);

  offsets_g = new Array<int>(NumOfGroups + 1);
  for (int k = 0; k <= NumOfGroups; k++) (*offsets_g)[k] = k * vfes_rad->GetNDofs();
  g_block = new BlockVector(*offsets_g);
  G = new ParGridFunction(vfes_rad, g_block->HostReadWrite());

  setGToZero();

  Variables_G.clear(), varNames_G.clear();
  for (int ig = 0; ig < NumOfGroups; ig++) {
    Variables_G.push_back(new ParGridFunction(fes_rad, G->HostReadWrite() + ig * fes_rad->GetNDofs()));
    varNames_G.push_back(std::string("G" + std::to_string(ig)));
  }

  P1Group = new P1Groups*[NumOfGroups];
  for (int ig = 0; ig < NumOfGroups; ig++) {
    P1Group[ig] = new P1Groups(mesh,fec_rad,fes_rad,dfes_rad,inputs,rbc_a_val,tableHost,axisymmetric,
                               Variables_G[ig],Radius,OneOverRadius);
    P1Group[ig]->setGroupID(ig); 
    P1Group[ig]->setNumOfGroups(NumOfGroups); 
    P1Group[ig]->readDataAndSetLinearTables(); 
  } 

  readTabulatedData();
  initializeInterpolationData();

  // Paraview setup
  iter = 0, time = 0.0;
  if (visualization){
    // visualization variables
    energySink_NEC = new ParGridFunction(fes_rad), *energySink_NEC = 0.0;
    energySink_NEC_tps = new ParGridFunction(fes_rad), *energySink_NEC_tps = 0.0;
    energySinkB2B = new ParGridFunction(fes_rad), *energySinkB2B = 0.0;
    energySinkOptThin = new ParGridFunction(fes_rad), *energySinkOptThin = 0.0;
    energySinkContinuum = new ParGridFunction(fes_rad), *energySinkContinuum = 0.0;
    G_integratedRadiance = new ParGridFunction(fes_rad), *G_integratedRadiance = 0.0;


    paraviewColl = new ParaViewDataCollection("P1model", mesh);
    paraviewColl->SetPrefixPath("ParaView");
    paraviewColl->SetLevelsOfDetail(order);
    paraviewColl->SetDataFormat(VTKFormat::BINARY);
    paraviewColl->SetHighOrderOutput(true);
    paraviewColl->SetPrecision(8);
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);


    paraviewColl->RegisterField("Temperature",Temperature);
    paraviewColl->RegisterField("energySink_P1",energySink_P1);
    paraviewColl->RegisterField("energySink_NEC",energySink_NEC);
    paraviewColl->RegisterField("energySink_NEC_tps",energySink_NEC_tps);

    paraviewColl->RegisterField("W_B2B",energySinkB2B);
    paraviewColl->RegisterField("W_OptThin",energySinkOptThin);
    paraviewColl->RegisterField("W_Continuum",energySinkContinuum);
    paraviewColl->RegisterField("W_OptIntermediate",energySinkOptIntermediate);

    paraviewColl->RegisterField("G_integratedRadiance",G_integratedRadiance);


    for (int var = 0; var < int(Variables_G.size()); var++) {
      paraviewColl->RegisterField(varNames_G[var], Variables_G[var]);
    }
    // paraviewColl->SetOwnData(true);
  }
}

P1Model::~P1Model() {

  delete [] P1Group;

  delete G;
  delete offsets_g;
  delete g_block;

  delete linearTableOpThin;
  delete linearTableContinuum;

  if (visualization){
    delete linearTableNEC;
    delete linearTableNEC_tps;
  }

  delete coordsDof;
  delete Radius;
  delete OneOverRadius;
}



void P1Model::readTabulatedData() {

  std::string filename,groupName,datasetName;
  std::string type, inputPath; 

  // Read table data for the contribution of the Optically Thin transitions;
  TableInputOpThin.xLogScale = inputs.P1TableInput.xLogScale;
  TableInputOpThin.fLogScale = inputs.P1TableInput.fLogScale;
  TableInputOpThin.order = inputs.P1TableInput.order;
  type = "P1Model"; 
  inputPath = type + "/";
  filename = "OpticallyThinContribution.h5";
  filename = inputs.pathToFiles + inputPath  + filename;
  datasetName = "table"; 
  readTable(filename, groupName, datasetName ,TableInputOpThin);
  linearTableOpThin = new LinearTable(TableInputOpThin);

  // Read table data for the contribution of continuum radiation;
  TableInputContinuum.xLogScale = inputs.P1TableInput.xLogScale;
  TableInputContinuum.fLogScale = inputs.P1TableInput.fLogScale;
  TableInputContinuum.order = inputs.P1TableInput.order;
  type = "P1Model"; 
  inputPath = type + "/";
  filename = "ContinuumContribution.h5";
  filename = inputs.pathToFiles + inputPath  + filename;
  datasetName = "table"; 
  readTable(filename, groupName, datasetName ,TableInputContinuum);
  linearTableContinuum = new LinearTable(TableInputContinuum);

  if (visualization){
    // Read NEC table data for comparison porpuses;
    TableInputNEC.xLogScale = inputs.P1TableInput.xLogScale;
    TableInputNEC.fLogScale = inputs.P1TableInput.fLogScale;
    TableInputNEC.order = inputs.P1TableInput.order;
    type = "NEC"; 
    inputPath = type + "/";
    filename = "nec_R.h5";
    filename = inputs.pathToFiles + inputPath  + filename;
    datasetName = "table"; 
    readTable(filename, groupName, datasetName ,TableInputNEC);
    linearTableNEC = new LinearTable(TableInputNEC);

    // Read NEC table data used in TPS sims for comparison porpuses;
    TableInputNEC_tps.xLogScale = inputs.P1TableInput.xLogScale;
    TableInputNEC_tps.fLogScale = inputs.P1TableInput.fLogScale;
    TableInputNEC_tps.order = inputs.P1TableInput.order;
    type = "NEC"; 
    inputPath = type + "/";
    filename = "nec_sample.0.h5";
    // filename = "nec_sample.1.h5";
    // filename = "nec_sample.2.h5";
    filename = inputs.pathToFiles + inputPath  + filename;
    datasetName = "table"; 
    readTable(filename, groupName, datasetName ,TableInputNEC_tps);
    linearTableNEC_tps = new LinearTable(TableInputNEC_tps); 
  }
}

void P1Model::initializeInterpolationData() {
  interp_flow_to_rad_ = new FieldInterpolator(mesh, fec, fes, mesh, fec_rad, fes_rad);
  interp_rad_to_flow_ = new FieldInterpolator(mesh, fec_rad, fes_rad, mesh, fec, fes);
  // interp_rad_to_flow_->setFromDGToH1Space(false);
}

void P1Model::Solve(ParGridFunction *src_Temperature, ParGridFunction *src_energySinkRad)
{

  double *dataG = G->HostReadWrite();

  interp_flow_to_rad_->PerformInterpolation(src_Temperature,Temperature);

  for (int ig = 0; ig < NumOfGroups; ig++) {
    if (rank0_){
      std::cout << " " << std::endl;
      std::cout << "**********************************************" << std::endl;
      std::cout << "      Solving P1 equation for group: " << ig << std::endl;
      std::cout << "**********************************************" << std::endl;
    }
    P1Group[ig]->Solve(Temperature);
  } 

  // Calculate the energy source term.
  for (int ig = 0; ig < NumOfGroups; ig++) {
    for (int i = 0; i < dof; i++) {
      (*energySinkOptIntermediate)[i] =  (*energySinkOptIntermediate)[i] + P1Group[ig]->getValueEnergySink(i); 
    }
  } 

  // Add the contribution of the Optically Thin transitions. Calculate NEC source term for comparison.
  for (int i = 0; i < dof; i++) {
    const double Th = (*Temperature)[i];
    double W_OptThin = computeOptThinRadContribution(Th);
    double W_Continuum = computeContinuumRadContribution(Th);
    double W_OptIntermediate = (*energySinkOptIntermediate)[i];

    (*energySink_P1)[i] =  W_OptIntermediate + W_OptThin + W_Continuum;
  }

  interp_rad_to_flow_->PerformInterpolation(energySink_P1,src_energySinkRad);


  // src_energySinkRad can  in some cases be positive. This means that these regions 
  // would be "warmed-up" due to heat transfer from radiation
  // const int nnodes = src_energySinkRad->ParFESpace()->GetNDofs();
  // for (int n = 0; n < nnodes; n++) {
  //   (*src_energySinkRad)[n] = min((*src_energySinkRad)[n], 0.0);
  // }

  if (visualization){
    G->HostRead();

    // Calculate/Assemble the energy source term.
    for (int ig = 0; ig < NumOfGroups; ig++) {
      for (int i = 0; i < dof; i++) {
        (*G_integratedRadiance)[i] =  (*G_integratedRadiance)[i] + dataG[i + ig * dof]; 
      }
    } 

    void findValuesOnWall();

    for (int i = 0; i < dof; i++) {
      const double Th = (*Temperature)[i];
      double W_OptThin = computeOptThinRadContribution(Th);
      double W_Continuum = computeContinuumRadContribution(Th);
      double W_OptIntermediate = (*energySinkOptIntermediate)[i];

      (*energySinkOptThin)[i] = W_OptThin;
      (*energySinkContinuum)[i] = W_Continuum;
      (*energySinkB2B)[i] = W_OptIntermediate + W_OptThin;
      (*energySink_NEC)[i] = computeEnergySinkNEC(Th);
      (*energySink_NEC_tps)[i] = computeEnergySinkNEC_tps(Th);
    }
    // Save data in the ParaView format
    paraviewColl->SetCycle(iter);
    paraviewColl->SetTime(time);
    paraviewColl->Save();
  }
}

void P1Model::setGToZero() {
  double *dataG = G->HostWrite();
  for (int i = 0; i < dof; i++) {
    for (int ig = 0; ig < NumOfGroups; ig++) {
      dataG[i + ig * dof] = 0.;
    }
  }
}

// I want function readTable being used in both P1Model and P1Groups classes! Is there a better way to do that? 
void P1Model::readTable(const std::string &filename, const std::string  &groupName, const std::string &datasetName, TableInput &result) {

  const int tableIndex = tableHost.size();
  tableHost.push_back(DenseMatrix());

  int Ndata;
  Array<int> dims(2);
  bool success = false;

  if (rank0_) {
    success = h5ReadTable(filename, groupName, datasetName, tableHost[tableIndex], dims); 

    // TODO(kevin): extend for multi-column array?
    Ndata = dims[0];
  }
  MPI_Bcast(&success, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
  if (!success) exit(ERROR);

  int *d_dims = dims.GetData();
  MPI_Bcast(&Ndata, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(d_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  assert(dims[0] > 0);
  assert(dims[1] == 2);

  if (!rank0_) tableHost[tableIndex].SetSize(dims[0], dims[1]);
  double *d_table = tableHost[tableIndex].GetData();
  MPI_Bcast(d_table, dims[0] * dims[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  result.Ndata = Ndata;
  result.xdata = tableHost[tableIndex].Read();
  result.fdata = tableHost[tableIndex].Read() + Ndata;

  return;
}

void P1Model::findValuesOnWall() {

  double *dataG = G->HostReadWrite();

  Array<int> wall_tdof_list; 
  Array<int> wall_bdr;

  wall_bdr.SetSize(mesh->bdr_attributes.Max());
  wall_bdr = 0;
  wall_bdr[7] = 1; // # Torch outer wall -> 7 patch = 8 -> Wall 

  if (mesh->GetNodes() == NULL) mesh->SetCurvature(1);

  fes_rad->FiniteElementSpace::GetEssentialTrueDofs(wall_bdr, wall_tdof_list);
  int wdof = wall_tdof_list.Size();

  int tot_wdof = 0;
  MPI_Allreduce(&wdof, &tot_wdof, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Gather local sizes on the root process
  std::vector<int> all_sizes(nprocs_);
  MPI_Gather(&wdof, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displacements(nprocs_, 0);
  if ( rank_ == 0) { 
    // Compute displacements for each process's data in the gathered array
    displacements[0] = 0;
    for (int i = 1; i < nprocs_; ++i) {
      displacements[i] = displacements[i - 1] + all_sizes[i - 1];
    }
  }
  // MPI_Bcast(all_sizes.data(), nprocs_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Bcast(displacements.data(), nprocs_, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Calculate the total size of the gathered data
  if ( rank_ == 0) {
    int total_size2 = 0;
    for (int i = 0; i < nprocs_; ++i) total_size2 += all_sizes[i];
    assert(tot_wdof == total_size2);
  }

  // Get coordinates
  ParGridFunction *x_coord = new ParGridFunction(fes_rad);
  ParGridFunction *y_coord = new ParGridFunction(fes_rad);

  *x_coord = 0.0;
  *y_coord = 0.0;

  for (int i = 0; i < fes_rad->GetNDofs(); i++) {
    Vector xyz(3);
    for (int d = 0; d < dim; d++) xyz[d] = (*coordsDof)[i + d * dof]; 
    (*x_coord)[i] = xyz[0];
    (*y_coord)[i] = xyz[1];
  }

  // Prepare variables tgo be written 
  std::vector<ParGridFunction *> gfVars;
  std::vector<std::string> gfVarNames;

  gfVars.clear(), gfVarNames.clear();
  gfVars.push_back(x_coord), gfVarNames.push_back(std::string("x_coord"));
  gfVars.push_back(y_coord), gfVarNames.push_back(std::string("y_coord"));
  gfVars.push_back(Temperature), gfVarNames.push_back(std::string("Temperature"));
  gfVars.push_back(G_integratedRadiance), gfVarNames.push_back(std::string("G_tot"));

  int nVars = gfVars.size();

  // Prepare file to be written
  string fileName, groupName;
  hid_t file = -1;
  hid_t data_soln;
  herr_t status;
  hsize_t dims[1];
  hid_t group = -1;
  hid_t dataspace = -1; 

  if (rank0_) {
    fileName = "./wall_data.h5";
    groupName = "torch_wall";

    file = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0);

    h5_save_attribute(file, "NumOfVars", nVars);
    h5_save_attribute(file, "NumOfWallData", tot_wdof);
    
    dims[0] = tot_wdof;

    dataspace = H5Screate_simple(1, dims, NULL);
    assert(dataspace >= 0);

    group = H5Gcreate(file, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(group >= 0);    
  }

  for (int iVar= 0; iVar < nVars; iVar++) {
    ParGridFunction *gf = gfVars[iVar];
    gf->GetTrueDofs();
   
    // Get local data
    std::vector<double> local_data(wdof, 0.0);

    for (int iwall = 0; iwall < wdof; iwall++) {
      int i = wall_tdof_list[iwall];
      local_data[iwall] = (*gf)[i];
    }

    std::vector<double> gathered_data(tot_wdof,0);
    // Use MPI_Gatherv to gather data from all processes
    MPI_Gatherv(local_data.data(), wdof, MPI_DOUBLE,gathered_data.data(), all_sizes.data(), 
                displacements.data(), MPI_DOUBLE,0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank0_) {
      string varName = gfVarNames[iVar];
      double *data = gathered_data.data();

      data_soln = H5Dcreate2(group, varName.c_str(), H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(data_soln >= 0);

      status = H5Dwrite(data_soln, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
      assert(status >= 0);
      H5Dclose(data_soln);
    }
  }

  if (rank0_) {
    if (group >= 0) H5Gclose(group);
    if (dataspace >= 0) H5Sclose(dataspace);
    if (file >= 0) H5Fclose(file);
  }
  
}

P1Groups::P1Groups(ParMesh *_mesh, FiniteElementCollection *_fec, 
                   ParFiniteElementSpace *_fes, ParFiniteElementSpace *_dfes,
                   const RadiationInput &_inputs,double _rbc_a_val, 
                   std::vector<mfem::DenseMatrix> &_tableHost, bool _axisymmetric, 
                   ParGridFunction *_G, ParGridFunction *_Radius, ParGridFunction *_OneOverRadius)
    :mesh(_mesh),
     fec_rad(_fec),
     fes_rad(_fes),
     dfes_rad(_dfes),
     inputs(_inputs),
     rbc_a_val(_rbc_a_val),
     tableHost(_tableHost),
     axisymmetric(_axisymmetric),
     g(_G),
     Radius(_Radius),
     OneOverRadius(_OneOverRadius)
    { 


  nprocs_ = Mpi::WorldSize();
  rank_ = Mpi::WorldRank();
  rank0_ = Mpi::Root();

  // Set some solver parameters
  MaxIters = inputs.MaxIters;
  P1_Tolerance = inputs.P1_Tolerance;
  PrintLevels = inputs.PrintLevels;

  dim = mesh->Dimension();
  dof = fes_rad->GetNDofs();

  total_num_dofs = fes_rad->GlobalTrueVSize();
  fes_rad->GetBoundaryTrueDofs(boundary_dofs);   // Extract the list of all the boundary DOFs. 

  // Boundary Conditions
  dbc_bdr = new Array<int>(mesh->bdr_attributes.Max());
  nbc_bdr = new Array<int>(mesh->bdr_attributes.Max());
  rbc_bdr = new Array<int>(mesh->bdr_attributes.Max());

  *dbc_bdr = 0, *nbc_bdr = 0, *rbc_bdr = 0;

  // This BC are for the torch case. 
  // # empty            -> 0 patch = 1 -> Axisymmetric -> dG/dn = 0       -> Neumann BC 
  // # Axis             -> 1 patch = 2 -> Axisymmetric -> dG/dn = 0       -> Neumann BC 
  // # Bottom wall      -> 2 patch = 3 -> Wall         -> dG/dn + a G = b -> Robin BC 
  // # Inlet            -> 3 patch = 4 -> Inlet        -> G = 0           -> Dirichlet BC 
  // # Jet "wall"       -> 4 patch = 5 -> Wall         -> dG/dn + a G = b -> Robin BC 
  // # Outlet           -> 5 patch = 6 -> Outlet       -> dG/dn = 0       -> Neumann BC 
  // # Top wall         -> 6 patch = 7 -> Wall         -> dG/dn + a G = b -> Robin BC 
  // # Torch outer wall -> 7 patch = 8 -> Wall         -> dG/dn + a G = b -> Robin BC 

  (*nbc_bdr)[0] = 1;
  if (axisymmetric) (*nbc_bdr)[1] = 1;
  (*rbc_bdr)[2] = 1;
  // (*dbc_bdr)[3] = 1;
  (*nbc_bdr)[3] = 1;
  (*rbc_bdr)[4] = 1;
  (*nbc_bdr)[5] = 1;
  (*rbc_bdr)[6] = 1;
  (*rbc_bdr)[7] = 1;

  // For a continuous basis the linear system must be modified to enforce an
  // essential (Dirichlet) boundary condition. In the DG case this is not
  // necessary as the boundary condition will only be enforced weakly.
  fes_rad->GetEssentialTrueDofs(*dbc_bdr, ess_tdof_list);

  AbsorptionCoef = new ParGridFunction(fes_rad);                                            
  oneOver3Absorp = new ParGridFunction(fes_rad); 
  emittedRad = new ParGridFunction(fes_rad);
  *AbsorptionCoef = 0.0, *oneOver3Absorp = 0.0, *emittedRad = 1.0;

  b = new ParLinearForm(fes_rad);
  a = new ParBilinearForm(fes_rad);

  if (isAxisymmetric()){ 
    RadiusCoeff = new GridFunctionCoefficient(Radius);
    OneOverRadiusCoeff = new GridFunctionCoefficient(OneOverRadius);
  }


  //     Define and apply a parallel PCG solver for AX=B with the BoomerAMG
  //     preconditioner from hypre. Extract the parallel grid function x
  //     corresponding to the finite element approximation X. This is the local
  //     solution on each processor.
  amg = new HypreBoomerAMG(A);
  pcg = new HyprePCG(_fes->GetComm());
  pcg->SetTol(P1_Tolerance);
  pcg->SetMaxIter(MaxIters);
  pcg->SetPrintLevel(PrintLevels);
  pcg->SetPreconditioner(*amg);

  // G_prec = new HypreSmoother();
  // G_solver = new CGSolver(_fes->GetComm());
  // G_solver->iterative_mode = false;
  // G_solver->SetRelTol(P1_Tolerance); // const double rel_tol = 1e-8;
  // G_solver->SetAbsTol(0.0);
  // G_solver->SetMaxIter(MaxIters);
  // G_solver->SetPrintLevel(PrintLevels);
  // G_prec->SetType(HypreSmoother::Jacobi);
  // G_solver->SetPreconditioner(*G_prec);
}

void P1Groups::readTable(const std::string &filename, const std::string  &groupName, const std::string &datasetName, TableInput &result) {

  const int tableIndex = tableHost.size();
  tableHost.push_back(DenseMatrix());

  int Ndata;
  Array<int> dims(2);
  bool success = false;

  if (rank0_) {
    success = h5ReadTable(filename, groupName, datasetName, tableHost[tableIndex], dims); 

    // TODO(kevin): extend for multi-column array?
    Ndata = dims[0];
  }
  MPI_Bcast(&success, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
  if (!success) exit(ERROR);

  int *d_dims = dims.GetData();
  MPI_Bcast(&Ndata, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(d_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
  assert(dims[0] > 0);
  assert(dims[1] == 2);

  if (!rank0_) tableHost[tableIndex].SetSize(dims[0], dims[1]);
  double *d_table = tableHost[tableIndex].GetData();
  MPI_Bcast(d_table, dims[0] * dims[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  result.Ndata = Ndata;
  result.xdata = tableHost[tableIndex].Read();
  result.fdata = tableHost[tableIndex].Read() + Ndata;

  return;
}


void P1Groups::readDataAndSetLinearTables() { 

  // Read Input table data and set LinearTable classes !!
  TableInputAbsorption.xLogScale = inputs.P1TableInput.xLogScale;
  TableInputAbsorption.fLogScale = inputs.P1TableInput.fLogScale;
  TableInputAbsorption.order = inputs.P1TableInput.order;

  TableInputEmittedRad.xLogScale = inputs.P1TableInput.xLogScale;
  TableInputEmittedRad.fLogScale = inputs.P1TableInput.fLogScale;
  TableInputEmittedRad.order = inputs.P1TableInput.order;


  std::string type("P1Model"); //  std::string type("NEC");
  std::string caseGroups("Groups21");
  std::string inputPath(inputs.pathToFiles + type + "/");

  std::string filename,groupName,datasetName;

  filename = inputPath + caseGroups + ".h5";
  groupName = "group"+ std::to_string(GroupID);

  datasetName = "Absorption"; 
  readTable(filename, groupName, datasetName ,TableInputAbsorption);
  linearTableAbsorption = new LinearTable(TableInputAbsorption);

  datasetName = "EmittedRad"; 
  readTable(filename, groupName, datasetName ,TableInputEmittedRad);   
  linearTableEmittedRad = new LinearTable(TableInputEmittedRad);

  return;
}



void P1Groups::Solve(ParGridFunction *_Temperature) {
  Temperature = _Temperature;

  //    Set up the coefficients
  // ConstantCoefficient one(1.0);
  ConstantCoefficient dbcCoef(dbc_val);
  ConstantCoefficient nbcCoef(nbc_val);

  // Calculate the coefficients which are functions of temperature.
  CalcTemperatureDependentParameters();

  GridFunctionCoefficient f_coeff (emittedRad); 
  GridFunctionCoefficient k_coeff (oneOver3Absorp);
  GridFunctionCoefficient b_coeff (AbsorptionCoef);

  ConstantCoefficient rbcACoef(rbc_a_val);
  rbcBCoef = &f_coeff; // How do I set the values for rbcBCoef on the boundaries?

  PowerCoefficient OneOverb_coeff(b_coeff,-1);
  ProductCoefficient rbcBCoef_temp(*rbcBCoef, OneOverb_coeff); // ???
  ProductCoefficient m_rbcBCoef(rbcACoef, rbcBCoef_temp); // ???


  if (isAxisymmetric()){ 
    rf_coeff = new ProductCoefficient(*RadiusCoeff, f_coeff);
    rk_coeff = new ProductCoefficient (*RadiusCoeff, k_coeff);
    rb_coeff = new ProductCoefficient (*RadiusCoeff, b_coeff);
    rm_rbcBCoef = new ProductCoefficient (*RadiusCoeff, m_rbcBCoef);
    r_rbcACoef = new ProductCoefficient (*RadiusCoeff,rbcACoef);
  } else {
    rf_coeff = &f_coeff;
    rk_coeff =  &k_coeff;
    rb_coeff = &b_coeff;
    rm_rbcBCoef = &m_rbcBCoef;
    r_rbcACoef = &rbcACoef;
  }    

  // Set the Dirichlet values in the solution vector
  // g->ProjectBdrCoefficient(dbcCoef, *dbc_bdr); 

  //    Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  b->Update();
  b->AddDomainIntegrator(new DomainLFIntegrator(*rf_coeff)); 

  // Add the desired value for n.Grad(u) on the Neumann boundary
  b->AddBoundaryIntegrator(new BoundaryLFIntegrator(nbcCoef), *nbc_bdr); 

  // // Add the desired value for n.Grad(u) + a*u on the Robin boundary
  b->AddBdrFaceIntegrator(new BoundaryLFIntegrator(*rm_rbcBCoef),*rbc_bdr);  // How do I assigne values to rbcBCoef?? This is for boundary but I evaluate at the domain!!
  b->Assemble(); 

  //    Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    and Mass domain integrators.
  a->Update();
  a->AddDomainIntegrator(new DiffusionIntegrator(*rk_coeff));
  a->AddDomainIntegrator(new MassIntegrator(*rb_coeff));

  //     Add a Mass integrator on the Robin boundary
  a->AddBoundaryIntegrator(new MassIntegrator(*r_rbcACoef), *rbc_bdr);
  a->Assemble();

  //     Form the linear system A X = B. This includes eliminating boundary
  //     conditions, applying AMR constraints, parallel assembly, etc.
  a->FormLinearSystem(ess_tdof_list, *g, *b, A, X, B); // ess_tdof_list  remains empty for pure Robin b.c.

  //     Define and apply a parallel PCG solver for AX=B with the BoomerAMG
  //     preconditioner from hypre. Extract the parallel grid function x
  //     corresponding to the finite element approximation X. This is the local
  //     solution on each processor.   
  // amg->SetOperator(A); // NOTE(malamast): not sure if we need this call.
  pcg->SetOperator(A); 
  pcg->Mult(B, X);

  // G_solver->SetOperator(A); 
  // G_solver->Mult(B, X);

  //     Recover the solution x as a grid function and save to file. The output
  //     can be viewed using GLVis as follows: "glvis -np <np> -m mesh -g sol"
  a->RecoverFEMSolution(X, *b, *g);

  return;
}



P1Groups::~P1Groups() {

  delete Temperature;

  delete AbsorptionCoef;
  delete emittedRad;
  delete oneOver3Absorp;

  delete[] rbc_bdr;

  delete b;
  delete a;

  delete linearTableAbsorption;
  delete linearTableEmittedRad;

  delete G_solver;
  delete G_prec;

  delete pcg; 
  delete amg;
}

void P1Groups::CalcTemperatureDependentParameters() {
  const double oneOverThree = 1.0/3.0;   
  for (int i = 0; i < dof; i++) {
    double Th = (*Temperature)[i];
    double kappa = linearTableAbsorption->eval(Th);
    (*AbsorptionCoef)[i] = kappa;
    (*emittedRad)[i] = linearTableEmittedRad->eval(Th);
    (*oneOver3Absorp)[i] = oneOverThree / (kappa+small);      
    // (*oneOver3Absorp)[i] = oneOverThree / (kappa+eps);      
  }
}




