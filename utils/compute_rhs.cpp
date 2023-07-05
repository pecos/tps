/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  mfem::Mpi::Init(argc, argv);
  TPS::Tps tps(MPI_COMM_WORLD);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  std::string basepath("utils/compute_rhs");
#ifdef HAVE_MASA
  std::string filename;
  tps.getRequiredInput((basepath + "/filename").c_str(), filename);
#endif

  M2ulPhyS *srcField = new M2ulPhyS( tps.getInputFilename(), &tps);

#ifdef HAVE_MASA
  RunConfiguration& srcConfig = srcField->GetConfig();
#endif

  GasMixture *mixture = srcField->getMixture();
  // int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  // Get meshes
  ParMesh* mesh = srcField->GetMesh();

  const int dim = mesh->Dimension();
  int numVariables = num_equation - dim + 1;

  if (mesh->GetNodes() == NULL) { mesh->SetCurvature(1); }
  // const int mesh_poly_deg =
  //   mesh->GetNodes()->FESpace()->GetElementOrder(0);
  cout << "Source mesh curvature: "
       << mesh->GetNodes()->OwnFEC()->Name() << endl;

  srcField->updatePrimitives();

  // 2) Set up source field
  FiniteElementCollection *src_fec = srcField->GetFEC();
  ParFiniteElementSpace *src_fes = srcField->GetFESpace();
  ParFiniteElementSpace *scalar_fes = srcField->GetScalarFES();
  ParFiniteElementSpace *vector_fes = srcField->GetVectorFES();
  const int nDofs = scalar_fes->GetNDofs();

  std::cout << "Source FE collection: " << src_fec->Name() << std::endl;

  ParGridFunction *src_state = srcField->GetSolutionGF();
  ParGridFunction *src_prim = srcField->getPrimitiveGF();

  RHSoperator *rhsOperator = srcField->getRHSoperator();
  ParGridFunction *rhs = new ParGridFunction(*src_state);

  ParaViewDataCollection *paraviewColl = srcField->GetParaviewColl();

  std::vector<ParGridFunction *> visualizationVariables(4 * numVariables);
  for (int func = 0; func < 3; func++) {
    std::string varName;
    ParGridFunction *targetVar;
    switch (func) {
      case 0:
        varName = "U";
        targetVar = src_state;
        break;
      case 1:
        varName = "Up";
        targetVar = src_prim;
        break;
      case 2:
        varName = "Rhs";
        targetVar = rhs;
        break;
    }
    for (int var = 0; var < numVariables; var++) {
      int offset = (var < 2) ? var : var - 1 + dim;
      if (var == 1) {
        visualizationVariables[var + func * numVariables] =
          new ParGridFunction(vector_fes, targetVar->HostReadWrite() + offset * nDofs);
      } else {
        visualizationVariables[var + func * numVariables] =
          new ParGridFunction(scalar_fes, targetVar->HostReadWrite() + offset * nDofs);
      }

      paraviewColl->RegisterField(varName + std::to_string(var),
                                  visualizationVariables[var + func * numVariables]);
    }
  }

  ParFiniteElementSpace dfes(src_fes->GetParMesh(), src_fes->FEColl(), dim, Ordering::byNODES);
  ParGridFunction coordinates(&dfes);
  src_fes->GetParMesh()->GetNodes(coordinates);

  rhsOperator->Mult(*src_state, *rhs);

#ifdef HAVE_MASA
  ForcingTerms *masaForcing = rhsOperator->getForcingTerm(rhsOperator->getMasaForcingIndex());
  ParGridFunction *masaRhs = new ParGridFunction(*src_state);
  double *dataMF = masaRhs->HostReadWrite();
  for (int i = 0; i < nDofs; i++)
    for (int eq = 0; eq < num_equation; eq++) dataMF[i + eq * nDofs] = 0.0;
  masaForcing->updateTerms(*masaRhs);
  for (int i = 0; i < nDofs; i++)
    for (int eq = 0; eq < num_equation; eq++) dataMF[i + eq * nDofs] *= -1.0;

  for (int var = 0; var < numVariables; var++) {
    int offset = (var < 2) ? var : var - 1 + dim;
    if (var == 1) {
      visualizationVariables[var + 3 * numVariables] =
        new ParGridFunction(vector_fes, masaRhs->HostReadWrite() + offset * nDofs);
    } else {
      visualizationVariables[var + 3 * numVariables] =
        new ParGridFunction(scalar_fes, masaRhs->HostReadWrite() + offset * nDofs);
    }

    paraviewColl->RegisterField("MASA-Rhs" + std::to_string(var),
                                visualizationVariables[var + 3 * numVariables]);
  }

  // compute component relative error of rhs.
  Vector error(numVariables);
  ParGridFunction scalarZero(*visualizationVariables[0]), vectorZero(*visualizationVariables[1]);
  double *dataScalar = scalarZero.HostReadWrite();
  double *dataVector = vectorZero.HostReadWrite();
  for (int i = 0; i < nDofs; i++) {
    dataScalar[i] = 0.0;
    for (int d = 0; d < dim; d++) dataVector[i + d * nDofs] = 0.0;
  }
  VectorGridFunctionCoefficient scalarZeroCoeff(&scalarZero), vectorZeroCoeff(&vectorZero);
  if (srcConfig.mmsCompareRhs_) {
    double norm;
    for (int var = 0; var < numVariables; var++) {
      VectorGridFunctionCoefficient masaCoeff(visualizationVariables[var + 3 * numVariables]);
      error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, masaCoeff);
      if (var == 1) {
        norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
      } else {
        norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
      }
      error(var) /= norm;
    }
  } else {
    for (int var = 0; var < numVariables; var++) {
      double norm;
      if (var == 1) {
        error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
        norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, vectorZeroCoeff);
      } else {
        error(var) = visualizationVariables[var + 2 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
        norm = visualizationVariables[var + 3 * numVariables]->ComputeLpError(2, scalarZeroCoeff);
      }
      error(var) /= norm;
    }
  }

  int numElems;
  int localElems = mesh->GetNE();
  MPI_Allreduce(&localElems, &numElems, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  
  if (mfem::Mpi::Root()) {
    std::cout << numElems << ",\t";
    for (int var = 0; var < numVariables; var++) {
      std::cout << error(var) << ",\t";
    }
    std::cout << std::endl;

    ofstream fID;
    fID.open(filename, std::ios_base::app | std::ios_base::out);
    fID << numElems << "\t";
    for (int var = 0; var < numVariables; var++) fID << error(var) << "\t";
    fID << "\n";
    fID.close();
  }
#endif

  bool saveGrad;
  tps.getInput((basepath + "/save_grad").c_str(), saveGrad, false);
  if (saveGrad) {
    ParGridFunction *src_gradUp = srcField->getGradientGF();
    for (int d = 0; d < dim; d++) {
      std::string varName;
      switch (d) {
        case 0:
          varName = "gradUp_x";
          break;
        case 1:
          varName = "gradUp_y";
          break;
        case 2:
          varName = "gradUp_z";
          break;
      }
      for (int eq = 0; eq < num_equation; eq++) {
        visualizationVariables.push_back(new ParGridFunction(scalar_fes, src_gradUp->HostReadWrite() + eq * nDofs + d * num_equation * nDofs));
        paraviewColl->RegisterField(varName + std::to_string(eq), visualizationVariables.back());
      }
    }
  }

  bool saveFlux;
  tps.getInput((basepath + "/save_flux").c_str(), saveFlux, false);
  if (saveFlux) {
    DenseTensor *src_flux = rhsOperator->getFlux();
    for (int d = 0; d < dim; d++) {
      std::string varName;
      switch (d) {
        case 0:
          varName = "flux_x";
          break;
        case 1:
          varName = "flux_y";
          break;
        case 2:
          varName = "flux_z";
          break;
      }
      for (int eq = 0; eq < num_equation; eq++) {
        visualizationVariables.push_back(new ParGridFunction(scalar_fes, src_flux->HostReadWrite() + d * nDofs + eq * dim * nDofs));
        paraviewColl->RegisterField(varName + std::to_string(eq), visualizationVariables.back());
      }
    }
  }

  srcField->writeParaview(0, 0.0);

  return 0;
}
