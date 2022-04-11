/* A utility to transfer tps solutions from one mesh to another */

#include "../src/M2ulPhyS.hpp"
#include "mfem.hpp"
#include <fstream>

using namespace mfem;
using namespace std;

int main (int argc, char *argv[])
{
  TPS::Tps tps(argc, argv);
  tps.parseCommandLineArgs(argc, argv);
  tps.parseInput();
  tps.chooseDevices();

  M2ulPhyS *srcField = new M2ulPhyS(tps.getMPISession(), tps.getInputFilename(), &tps);
  RunConfiguration& srcConfig = srcField->GetConfig();

  GasMixture *mixture = srcField->getMixture();
  int numSpecies = mixture->GetNumSpecies();
  int num_equation = mixture->GetNumEquations();

  // Get meshes
  ParMesh* mesh = srcField->GetMesh();

  const int dim = mesh->Dimension();
  int numVariables = num_equation - dim + 1;

  if (mesh->GetNodes() == NULL) { mesh->SetCurvature(1); }
  const int mesh_poly_deg =
    mesh->GetNodes()->FESpace()->GetElementOrder(0);
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

  RHSoperator rhsOperator = srcField->getRHSoperator();
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

  rhsOperator.Mult(*src_state, *rhs);

#ifdef HAVE_MASA
  ForcingTerms *masaForcing = rhsOperator.getForcingTerm(rhsOperator.getMasaForcingIndex());
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
#endif

  srcField->writeParaview(0, 0.0);

  return 0;
}
