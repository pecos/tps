#include "BoundaryCondition.hpp"
#include "general/forall.hpp"

BoundaryCondition::BoundaryCondition(RiemannSolver* _rsolver, 
                                     EquationOfState* _eqState, 
                                     ParFiniteElementSpace *_vfes,
                                     IntegrationRules *_intRules,
                                     double &_dt,
                                     const int _dim, 
                                     const int _num_equation, 
                                     const int _patchNumber,
                                     const double _refLength ):
rsolver(_rsolver),
eqState(_eqState),
vfes(_vfes),
intRules(_intRules),
dt(_dt),
dim(_dim),
num_equation(_num_equation),
patchNumber(_patchNumber),
refLength(_refLength)
{
  BCinit = false;
}

BoundaryCondition::~BoundaryCondition()
{
}

double BoundaryCondition::aggregateArea(int bndry_patchnum, MPI_Comm bc_comm)
{
  double area = 0;

  // loop over boundary faces and accumulate area of matching patch
  for(int bel=0;bel<vfes->GetNBE(); bel++)
    {
      int attr = vfes->GetBdrAttribute(bel);
      if( attr == bndry_patchnum )
	{
	  FaceElementTransformations *Tr = vfes->GetMesh()->GetBdrFaceTransformations(bel);
	  const IntegrationRule &ir = IntRules.Get(Tr->GetGeometryType(), Tr->OrderJ());

	  for (int p = 0; p < ir.GetNPoints(); p++)
	    {
	      const IntegrationPoint &ip = ir.IntPoint(p);
	      area += Tr->Weight() * ip.weight;
	    }
	}
    }

  double areaTotal;
  MPI_Allreduce(&area, &areaTotal,1, MPI_DOUBLE, MPI_SUM, bc_comm);

  return(areaTotal);
}

int BoundaryCondition::aggregateBndryFaces(int bndry_patchnum, MPI_Comm bc_comm)
{
  int nfaces = 0;
  for(int bel=0;bel<vfes->GetNBE(); bel++)
    {
      int attr = vfes->GetBdrAttribute(bel);
      if( attr == bndry_patchnum )
	nfaces++;
    }
  int nfacesTotal;
  MPI_Allreduce(&nfaces, &nfacesTotal,1, MPI_INT, MPI_SUM, bc_comm);
  return(nfacesTotal);
}

void BoundaryCondition::setElementList(Array<int> &_listElems)
{
  listElems.SetSize( _listElems.Size() );
  for(int i=0;i<listElems.Size();i++) listElems[i] = _listElems[i];
  listElems.ReadWrite();
}

void BoundaryCondition::copyValues(const Vector& orig, Vector& target, const double& mult)
{
  const double *dOrig = orig.Read();
  double *dTarget = target.Write();
  
  MFEM_FORALL(i,target.Size(),
  {
    dTarget[i] = dOrig[i]*mult;
  });
}
