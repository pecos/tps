/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

Vector L_, Up0_;
Array<int> kx_;
DenseMatrix offset_, dUp_;
const double pi_ = atan(1.0) * 4.0;

double uniformRandomNumber() {
  return ((double) rand()) / (RAND_MAX);
}

void setRandomPrimitiveState(GasMixture *mixture) {
  const double numEquation = mixture->GetNumEquations();
  const double numSpecies = mixture->GetNumSpecies();
  const double numActiveSpecies = mixture->GetNumActiveSpecies();
  const bool ambipolar = mixture->IsAmbipolar();
  const bool twoTemperature = mixture->IsTwoTemperature();
  const double dim = mixture->GetDimension();
  const double nvel = mixture->GetNumVels();

  for (int d = 0; d < nvel; d++) {
    Up0_(d + 1) = -0.5 + 1.0 * uniformRandomNumber();
  }
  Up0_(nvel + 1) = 300.0 * ( 0.9 + 0.2 * uniformRandomNumber() );
  if (twoTemperature) Up0_(numEquation - 1) = 400.0 * ( 0.8 + 0.4 * uniformRandomNumber() );

  Vector n_sp(numSpecies);
  for (int sp = 0; sp < numSpecies; sp++){
    if (ambipolar && (sp == numSpecies - 2)) continue;
    n_sp(sp) = 1.0e0 * uniformRandomNumber();
    if (sp == numSpecies - 1) n_sp(sp) += 2.0e0;
  }
  if (ambipolar) {
    double ne = mixture->computeAmbipolarElectronNumberDensity(&n_sp[0]);
    n_sp(numSpecies - 2) = ne;
  }
  grvy_printf(GRVY_INFO, "\n number densities.\n");
  for (int sp = 0; sp < numSpecies; sp++){
    grvy_printf(GRVY_INFO, "%.8E, ", n_sp(sp));
  }
  grvy_printf(GRVY_INFO, "\n");

  double rho = 0.0;
  for (int sp = 0; sp < numSpecies; sp++) rho += n_sp(sp) * mixture->GetGasParams(sp,GasParams::SPECIES_MW);
  Up0_(0) = rho;

  for (int sp = 0; sp < numActiveSpecies; sp++) Up0_(nvel + 2 + sp) = n_sp(sp);

  // overall 10% variation in space
  for (int eq = 0; eq < numEquation; eq++)
    for (int d = 0; d < dim; d++) dUp_(eq, d) = 0.1 * Up0_(eq) / dim * uniformRandomNumber();
}

void exactPrimFunction(const Vector &x, double tin, Vector &y) {
  for (int eq = 0; eq < y.Size(); eq++) {
    y(eq) = Up0_(eq);
    for (int d = 0; d < x.Size(); d++)
      y(eq) += dUp_(eq, d) * sin(2.0 * pi_ * kx_[d] * (x(d) / L_(d) - offset_(eq, d)));
  }
}

void exactGradXFunction(const Vector &x, double tin, Vector &y) {
  y = 0.0;
  const int d = 0;
  for (int eq = 0; eq < y.Size(); eq++)
    y(eq) += dUp_(eq, d) * 2.0 * pi_ * kx_[d] / L_(d) * cos(2.0 * pi_ * kx_[d] * (x(d) / L_(d) - offset_(eq, d)));
}

void exactGradYFunction(const Vector &x, double tin, Vector &y) {
  y = 0.0;
  const int d = 1;
  for (int eq = 0; eq < y.Size(); eq++)
    y(eq) += dUp_(eq, d) * 2.0 * pi_ * kx_[d] / L_(d) * cos(2.0 * pi_ * kx_[d] * (x(d) / L_(d) - offset_(eq, d)));
}

void exactGradZFunction(const Vector &x, double tin, Vector &y) {
  y = 0.0;
  const int d = 2;
  for (int eq = 0; eq < y.Size(); eq++)
    y(eq) += dUp_(eq, d) * 2.0 * pi_ * kx_[d] / L_(d) * cos(2.0 * pi_ * kx_[d] * (x(d) / L_(d) - offset_(eq, d)));
}

int main (int argc, char *argv[])
{
  srand (time(NULL));

  TPS::Tps tps(argc, argv);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  std::string basepath("test/test_gradient");
  std::string filename;
  tps.getRequiredInput((basepath + "/filename").c_str(), filename);

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), tps.getInputFilename(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();

  GasMixture *mixture = srcField->getMixture();
  int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  // Get meshes
  ParMesh* mesh = srcField->GetMesh();

  const int dim = mesh->Dimension();
  Up0_.SetSize(num_equation);
  dUp_.SetSize(num_equation, dim);
  setRandomPrimitiveState(mixture);

  L_.SetSize(dim);
  kx_.SetSize(dim);
  offset_.SetSize(num_equation, dim);
  tps.getRequiredInput((basepath + "/domain_length/x").c_str(), L_[0]);
  tps.getRequiredInput((basepath + "/domain_length/y").c_str(), L_[1]);
  if (dim == 3) tps.getRequiredInput((basepath + "/domain_length/z").c_str(), L_[2]);
  tps.getInput((basepath + "/wavenumber/x").c_str(), kx_[0], 2);
  tps.getInput((basepath + "/wavenumber/y").c_str(), kx_[1], 2);
  if (dim == 3) tps.getInput((basepath + "/wavenumber/z").c_str(), kx_[2], 2);
  for (int eq = 0; eq < num_equation; eq++)
    for (int d = 0; d < dim; d++) offset_(eq, d) = uniformRandomNumber();

  if (mesh->GetNodes() == NULL) { mesh->SetCurvature(1); }
  const int mesh_poly_deg =
    mesh->GetNodes()->FESpace()->GetElementOrder(0);
  cout << "Source mesh curvature: "
       << mesh->GetNodes()->OwnFEC()->Name() << endl;

  ParFiniteElementSpace *src_fes = srcField->GetFESpace();
  const int nDofs = src_fes->GetNDofs();
  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

//   srcField->updatePrimitives();
//
//   // 2) Set up source field
//   FiniteElementCollection *src_fec = srcField->GetFEC();

  ParFiniteElementSpace *scalar_fes = srcField->GetScalarFES();
//   ParFiniteElementSpace *vector_fes = srcField->GetVectorFES();
//   const int nDofs = scalar_fes->GetNDofs();
//
//   std::cout << "Source FE collection: " << src_fec->Name() << std::endl;
//
//   ParGridFunction *src_state = srcField->GetSolutionGF();
  ParGridFunction *src_prim = srcField->getPrimitiveGF();
  ParGridFunction *src_grad = srcField->getGradientGF();

  VectorFunctionCoefficient prim0(num_equation, &(exactPrimFunction));
  prim0.SetTime(0.0);
  src_prim->ProjectCoefficient(prim0);

  RHSoperator *rhsOperator = srcField->getRHSoperator();
  Gradients *gradients = rhsOperator->getGradients();
  gradients->computeGradients();
//   ParGridFunction *rhs = new ParGridFunction(*src_state);

  ParaViewDataCollection *paraviewColl = srcField->GetParaviewColl();

  std::vector<ParGridFunction *> visualizationVariables((dim + 1) * num_equation);
  for (int func = 0; func < dim + 1; func++) {
    std::string varName;
    ParGridFunction *targetVar;
    switch (func) {
      case 0:
        varName = "Up";
        targetVar = src_prim;
        break;
      case 1:
        varName = "gradUp_x";
        targetVar = src_grad;
        break;
      case 2:
        varName = "gradUp_y";
        targetVar = src_grad;
        break;
      case 3:
        varName = "gradUp_z";
        targetVar = src_grad;
        break;
    }
    int offset = (func < 2) ? 0 : (func - 1);
    for (int eq = 0; eq < num_equation; eq++) {
      visualizationVariables[eq + func * num_equation] =
        new ParGridFunction(scalar_fes, targetVar->HostReadWrite() + eq * nDofs + offset * num_equation * nDofs);

      paraviewColl->RegisterField(varName + std::to_string(eq),
                                  visualizationVariables[eq + func * num_equation]);
    }
  }
//
//
//   rhsOperator->Mult(*src_state, *rhs);
//
// #ifdef HAVE_MASA
//   ForcingTerms *masaForcing = rhsOperator->getForcingTerm(rhsOperator->getMasaForcingIndex());
//   ParGridFunction *masaRhs = new ParGridFunction(*src_state);
//   double *dataMF = masaRhs->HostReadWrite();
//   for (int i = 0; i < nDofs; i++)
//     for (int eq = 0; eq < num_equation; eq++) dataMF[i + eq * nDofs] = 0.0;
//   masaForcing->updateTerms(*masaRhs);
//   for (int i = 0; i < nDofs; i++)
//     for (int eq = 0; eq < num_equation; eq++) dataMF[i + eq * nDofs] *= -1.0;
//
//   for (int var = 0; var < numVariables; var++) {
//     int offset = (var < 2) ? var : var - 1 + dim;
//     if (var == 1) {
//       visualizationVariables[var + 3 * numVariables] =
//         new ParGridFunction(vector_fes, masaRhs->HostReadWrite() + offset * nDofs);
//     } else {
//       visualizationVariables[var + 3 * numVariables] =
//         new ParGridFunction(scalar_fes, masaRhs->HostReadWrite() + offset * nDofs);
//     }
//
//     paraviewColl->RegisterField("MASA-Rhs" + std::to_string(var),
//                                 visualizationVariables[var + 3 * numVariables]);
//   }
//
//   // compute component relative error of rhs.
//   Vector error(numVariables);
//   ParGridFunction scalarZero(*visualizationVariables[0]), vectorZero(*visualizationVariables[1]);
//   double *dataScalar = scalarZero.HostReadWrite();
//   double *dataVector = vectorZero.HostReadWrite();
//   for (int i = 0; i < nDofs; i++) {
//     dataScalar[i] = 0.0;
//     for (int d = 0; d < dim; d++) dataVector[i + d * nDofs] = 0.0;
//   }
//   VectorGridFunctionCoefficient scalarZeroCoeff(&scalarZero), vectorZeroCoeff(&vectorZero);
//   if (srcConfig.mmsCompareRhs_) {
//     double norm;
//     for (int var = 0; var < numVariables; var++) {
//       VectorGridFunctionCoefficient masaCoeff(visualizationVariables[var + 3 * numVariables]);
//       error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, masaCoeff);
//       if (var == 1) {
//         norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
//       } else {
//         norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
//       }
//       error(var) /= norm;
//     }
//   } else {
//     for (int var = 0; var < numVariables; var++) {
//       double norm;
//       if (var == 1) {
//         error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
//         norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
//       } else {
//         error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
//         norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
//       }
//       error(var) /= norm;
//     }
//   }
//
//   int numElems;
//   int localElems = mesh->GetNE();
//   MPI_Allreduce(&localElems, &numElems, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//
//   mfem::MPI_Session *mpi = &(tps.getMPISession());
//   if (mpi->Root()) {
//     std::cout << numElems << ",\t";
//     for (int var = 0; var < numVariables; var++) {
//       std::cout << error(var) << ",\t";
//     }
//     std::cout << std::endl;
//
//     ofstream fID;
//     fID.open(filename, std::ios_base::app | std::ios_base::out);
//     fID << numElems << "\t";
//     for (int var = 0; var < numVariables; var++) fID << error(var) << "\t";
//     fID << "\n";
//     fID.close();
//   }
// #endif

  srcField->writeParaview(0, 0.0);

  return 0;
}
