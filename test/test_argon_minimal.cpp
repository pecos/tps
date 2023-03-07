/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

Array<int> readTable(const string fileName, const string datasetName, DenseMatrix &output) {
  hid_t file = -1;
  if (file_exists(fileName)) {
    file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  } else {
    grvy_printf(GRVY_ERROR, "[ERROR]: Unable to open file -> %s\n", fileName.c_str());
    exit(ERROR);
  }
  assert(file >= 0);

  hid_t datasetID, dataspace;
  datasetID = H5Dopen2(file, datasetName.c_str(), H5P_DEFAULT);
  assert(datasetID >= 0);
  dataspace = H5Dget_space(datasetID);
  const int ndims = H5Sget_simple_extent_ndims(dataspace);
  hsize_t dims[ndims];
  // int dummy = H5Sget_simple_extent_dims(dataspace,dims,NULL);
  H5Sget_simple_extent_dims(dataspace,dims,NULL);

  output.SetSize(dims[0], dims[1]);

  herr_t status;
  status = H5Dread(datasetID, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, output.HostReadWrite());
  assert(status >= 0);
  H5Dclose(datasetID);
  H5Fclose(file);

  Array<int> shape(2);
  shape[0] = dims[0];
  shape[1] = dims[1];

  return shape;
}

int main (int argc, char *argv[])
{
  TPS::Tps tps;
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  // srand (time(NULL));
  // double dummy = uniformRandomNumber();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), &tps);

  RunConfiguration& srcConfig = srcField->GetConfig();
  assert(srcConfig.GetWorkingFluid()==WorkingFluid::USER_DEFINED);
  assert(srcConfig.GetGasModel()==PERFECT_MIXTURE);
  assert((srcConfig.GetTranportModel()==ARGON_MINIMAL) || (srcConfig.GetTranportModel()==ARGON_MIXTURE));

  int dim = 3;

  Vector relErrorThreshold = Vector({2e-4, 2e-4, 3e-4});
  Vector absErrorThreshold = Vector({5e-8, 4e-5, 1.5e-4});
  // int nTrials = 10;

  double pressure = 101325.0;

  PerfectMixture *mixture = new PerfectMixture(srcConfig, dim, dim);
  TransportProperties *transport;
  switch (srcConfig.GetTranportModel()) {
    case ARGON_MINIMAL:
      transport = new ArgonMinimalTransport(mixture, srcConfig);
      break;
    case ARGON_MIXTURE:
      transport = new ArgonMixtureTransport(mixture, srcConfig);
      break;
  }
  //ArgonMinimalTransport *transport = new ArgonMinimalTransport(mixture, srcConfig);
  int numSpecies = mixture->GetNumSpecies();
  int numActiveSpecies = mixture->GetNumActiveSpecies();
  int num_equation = mixture->GetNumEquations();
  // std::map<std::string, int> *speciesMapping = mixture->getSpeciesMapping();
  int electronIndex = srcConfig.speciesMapping["E"];
  int ionIndex = srcConfig.speciesMapping["Ar.+1"];
  int neutralIndex = srcConfig.speciesMapping["Ar"];

  std::string fileName = "./ref_solns/transport/Chung.transport.h5";
  std::string datasetName = "transport";
  DenseMatrix refValues;
  Array<int> dims1 = readTable(fileName, datasetName, refValues);
  grvy_printf(GRVY_INFO, "\n T\t Viscosity \t Thermal cond. (H) \t Thermal cond. (E) \t Ion. degree \n");
  for (int i = 0; i < dims1[0]; i++) {
    for (int j = 0; j < dims1[1]; j++) {
      std::cout << refValues(i,j) << ",\t";
    }
    std::cout << std::endl;
  }
  fileName = "./ref_solns/transport/Devoto1973.transport.h5";
  DenseMatrix Devoto;
  Array<int> dims2 = readTable(fileName, datasetName, Devoto);

  Array<int> idxs(dims1[0]);
  idxs[0] = 2;
  idxs[1] = 3;
  idxs[2] = 4;
  idxs[3] = 5;
  idxs[4] = 6;
  idxs[5] = 7;
  idxs[6] = 9;
  idxs[7] = 11;
  idxs[8] = 13;
  idxs[9] = 15;
  idxs[10] = 17;

  Vector Tm(dims1[0]);
  refValues.GetColumn(0, Tm);
  Vector ionizationDegree(dims1[0]);
  refValues.GetColumn(dims1[1] - 1, ionizationDegree);
  // double error = 0.0, error1 = 0.0;
  Vector error(3);
  for (int i = 0; i < dims1[0]; i++) {
    double nTotal = pressure / UNIVERSALGASCONSTANT / Tm(i);
    double a = ionizationDegree(i);
    Vector n_sp(3);
    n_sp(electronIndex) = a / (1.0 + a) * nTotal;
    n_sp(ionIndex) = n_sp(electronIndex);
    n_sp(neutralIndex) = (1.0 - a) / (1.0 + a) * nTotal;
    // Vector X_sp(3);
    // std::cout << "X: ";
    // for (int sp = 0; sp < 3; sp++) {
    //   X_sp(sp) = n_sp(sp) / nTotal;
    //   std::cout << X_sp(sp) << ",\t";
    // }
    // std::cout << std::endl;

    double rho = 0.0;
    for (int sp = 0; sp < 3; sp++) rho += n_sp(sp) * mixture->GetGasParams(sp, GasParams::SPECIES_MW);

    Vector primitiveState(num_equation);
    primitiveState = 0.0;
    primitiveState(0) = rho;
    primitiveState(1 + dim) = Tm(i);
    for (int sp = 0; sp < numActiveSpecies; sp++) {
      primitiveState(2 + dim + sp) = n_sp(sp);
    }
    // for (int eq = 0; eq < num_equation; eq++) {
    //   std::cout << primitiveState(eq) << ",\t";
    // }
    // std::cout << n_sp(electronIndex) << ",\t" << n_sp(neutralIndex) << std::endl;

    Vector conservedState(num_equation);
    mixture->GetConservativesFromPrimitives(primitiveState, conservedState);

    DenseMatrix gradUp(num_equation,3);
    gradUp = 0.0;

    Vector Efield(3);
    Efield = 0.0;

    Vector transportBuffer;
    transportBuffer.SetSize(FluxTrns::NUM_FLUX_TRANS);
    DenseMatrix diffusionVelocity(numSpecies, dim);
    transport->ComputeFluxTransportProperties(conservedState, gradUp, Efield, transportBuffer, diffusionVelocity);

    double visc, bulkVisc, visc_vec[2];
    transport->GetViscosities(conservedState, primitiveState, visc_vec);
    visc = visc_vec[0];
    bulkVisc = visc_vec[1];

    double visc_err = abs(visc - transportBuffer[FluxTrns::VISCOSITY]);
    double bulk_visc_err = abs(bulkVisc - transportBuffer[FluxTrns::BULK_VISCOSITY]);
    double consistencyThreshold = 1.0e-18;
    if (visc_err > consistencyThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeFluxTransportProperties and GetViscosities are not consistent!");
      grvy_printf(GRVY_ERROR, "\n Viscosity Error: %.15E", visc_err);
      exit(ERROR);
    } else if (bulk_visc_err > consistencyThreshold) {
      grvy_printf(GRVY_ERROR, "\n ComputeFluxTransportProperties and GetViscosities are not consistent!");
      grvy_printf(GRVY_ERROR, "\n Bulk Viscosity Error: %.15E", bulk_visc_err);
      exit(ERROR);
    }

    // Check if the artificial multipliers are applied correctly.
    if (srcConfig.argonTransportInput.multiply) {
      for (int t = 0; t < FluxTrns::NUM_FLUX_TRANS; t++)
        transportBuffer[t] /= srcConfig.argonTransportInput.fluxTrnsMultiplier[t];
    }

    // error = max(error, abs(refValues(i,1) - transportBuffer[FluxTrns::VISCOSITY]) / refValues(i,1));
    error(0) = abs(refValues(i,1) - transportBuffer[FluxTrns::VISCOSITY]) / refValues(i,1);
    if (error(0) > relErrorThreshold(0)) {
      grvy_printf(GRVY_ERROR, "\n Viscosity deviates from the reference value.");
      grvy_printf(GRVY_ERROR, "\n Relative Error: %.15E", error(0));
      exit(ERROR);
    }
    error(1) = abs(refValues(i,2) - transportBuffer[FluxTrns::HEAVY_THERMAL_CONDUCTIVITY]) / refValues(i,2);
    if (error(1) > relErrorThreshold(1)) {
      grvy_printf(GRVY_ERROR, "\n Heavy-species thermal conductivity deviates from the reference value.");
      grvy_printf(GRVY_ERROR, "\n Relative Error: %.15E", error(1));
      exit(ERROR);
    }
    if (i < 4) {
      error(2) = abs(refValues(i,3) - transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY]);
      if (error(2) > absErrorThreshold(2)) {
        grvy_printf(GRVY_ERROR, "\n Heavy-species thermal conductivity deviates from the reference value.");
        grvy_printf(GRVY_ERROR, "\n Absolute Error: %.15E", error(1));
        exit(ERROR);
      }
    } else {
      error(2) = abs(refValues(i,3) - transportBuffer[FluxTrns::ELECTRON_THERMAL_CONDUCTIVITY]) / refValues(i,3);
      if (error(2) > relErrorThreshold(2)) {
        grvy_printf(GRVY_ERROR, "\n Heavy-species thermal conductivity deviates from the reference value.");
        grvy_printf(GRVY_ERROR, "\n Relative Error: %.15E", error(1));
        exit(ERROR);
      }
    }

  } // for i

  grvy_printf(GRVY_INFO, "\n PASS: argon ternary mixture transport. \n");

  return 0;
}
