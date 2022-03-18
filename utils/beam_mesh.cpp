// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//          -------------------------------------------------------
//          Twist Miniapp:  Generate simple twisted periodic meshes
//          -------------------------------------------------------
//
// This miniapp generates simple periodic meshes to demonstrate MFEM's handling
// of periodic domains. MFEM's strategy is to use a discontinuous vector field
// to define the mesh coordinates on a topologically periodic mesh. It works by
// defining a stack of individual elements and stitching together the top and
// bottom of the mesh. The stack can also be twisted so that the vertices of the
// bottom and top can be joined with any integer offset (for tetrahedral and
// wedge meshes only even offsets are supported).
//
// Compile with: make twist
//
// Sample runs:  twist
//               twist -no-pm
//               twist -nt -2 -no-pm
//               twist -nt 2 -e 4
//               twist -nt 2 -e 6
//               twist -nt 3 -e 8
//
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

static Element::Type el_type_ = Element::QUADRILATERAL;
static int    order_ = 3;
static int    nx_    = 5;
static int    nt_    = 3;
static double a_     = 5.0;
static double b_     = 1.0;
// static double c_     = 3.0;

// void pts(int iphi, int t, double x[]);
// void trans(const Vector &x, Vector &p);

template<typename T>
bool InArray(const T* begin, size_t sz, T i)
{
   const T *end = begin + sz;
   return std::find(begin, end, i) != end;
}

bool IndicesAreConnected(const Table &t, int i, int j)
{
   return InArray(t.GetRow(i), t.RowSize(i), j)
          && InArray(t.GetRow(j), t.RowSize(j), i);
}

int main(int argc, char *argv[])
{
   int ser_ref_levels = 3;
   int el_type = 4;
   bool per_mesh = true;
   bool dg_mesh  = true;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&nx_, "-nx", "--num-elements-x",
                  "Number of elements in x-direction.");
   args.AddOption(&nt_, "-nt", "--num-unit-elements",
                  "Unit number of elements in both directions.");
   args.AddOption(&order_, "-o", "--mesh-order",
                  "Order (polynomial degree) of the mesh elements.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&a_, "-a", "--base-x",
                  "Width of the base in x-direction.");
   args.AddOption(&b_, "-b", "--base-y",
                  "Width of the base in y-direction.");
   // args.AddOption(&c_, "-c", "--height",
   //                "Height in z-direction.");
   args.AddOption(&el_type, "-e", "--element-type",
                  "Element type: 4 - Quadrilateral, 6 - Triangle.");
   // args.AddOption(&per_mesh, "-pm", "--periodic-mesh",
   //                "-no-pm", "--non-periodic-mesh",
   //                "Enforce periodicity in z-direction "
   //                "(requires discontinuous mesh).");
   // args.AddOption(&dg_mesh, "-dm", "--discont-mesh", "-cm", "--cont-mesh",
   //                "Use discontinuous or continuous space for the mesh nodes.");
   // args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
   //                "--no-visualization",
   //                "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   if (nt_ < 3) {
     cout << "nt: " << nt_ << " is too small for mesh generation. Use nt >= 3." << endl;
   }

   // The output mesh could be tetrahedra, hexahedra, or prisms
   switch (el_type)
   {
      case 4:
         el_type_ = Element::QUADRILATERAL;
         break;
      case 6:
         el_type_ = Element::TRIANGLE;
         break;
      default:
         cout << "Unsupported element type" << endl;
         exit(1);
         break;
   }

   // Define the mesh
   std::cout << a_ << ", " << b_ << std::endl;
   Mesh orig_mesh = Mesh::MakeCartesian2D(nt_*nx_, nt_, el_type_, false, a_, b_, false);

   std::vector<int> v2v(orig_mesh.GetNV());
   std::vector<Vector> translations = {Vector({a_,0.0}), Vector({0.0,b_})};
   v2v = orig_mesh.CreatePeriodicVertexMapping(translations);
   Mesh mesh = Mesh::MakePeriodic(orig_mesh,v2v);
   std::cout << "created periodic mesh." << std::endl;
  //  assert(mesh.GetNV() == nx_ * 1);
  // if (el_type_ == Element::QUADRILATERAL)
  // {
  //    const Table &e2e = mesh.ElementToElementTable();
  //    // std::cout << e2e << std::endl;
  //    for (int i=0; i<nx_*nt_; i++)
  //    {
  //       // Bottom row connected to top row
  //       assert(IndicesAreConnected(e2e, i, i + nx_*nt_*(nt_-1)));
  //     }
  //     for (int i=0; i<nt_; i++){
  //       // Left column connected to right column
  //       assert(IndicesAreConnected(e2e, i*nx_*nt_, nx_*nt_-1 + i*nx_*nt_));
  //    }
  // }

  // Promote to high order mesh
  if (order_ > 1 || dg_mesh || per_mesh)
  {
     orig_mesh.SetCurvature(order_, dg_mesh || per_mesh, 2, Ordering::byVDIM);
  }

   // if (nt_ != 0 )
   // {
   //    mesh.Transform(trans);
   // }

   // while (per_mesh)
   // {
   //
   //    // Verify topological compatibility
   //    if (nt_ % 2 == 1 && (el_type_ == Element::TETRAHEDRON ||
   //                         el_type_ == Element::WEDGE))
   //    {
   //       cout << "Diagonal cuts on the base and top must line up "
   //            << "for a periodic mesh!" << endl;
   //       exit(1);
   //    }
   //
   //    int nnode = 4;
   //    int noff = (nt_ >= 0) ? 0 : (nnode * (1 - nt_ / nnode));
   //
   //    Array<int> v2v(mesh.GetNV());
   //    for (int i = 0; i < v2v.Size() - nnode; i++)
   //    {
   //       v2v[i] = i;
   //    }
   //    // identify vertices at the extremes of the stack
   //    switch ((noff + nt_) % nnode)
   //    {
   //       case 0:
   //          v2v[v2v.Size() - nnode + 0] = 0;
   //          v2v[v2v.Size() - nnode + 1] = 1;
   //          v2v[v2v.Size() - nnode + 2] = 2;
   //          v2v[v2v.Size() - nnode + 3] = 3;
   //          break;
   //       case 1:
   //          v2v[v2v.Size() - nnode + 0] = 2;
   //          v2v[v2v.Size() - nnode + 1] = 0;
   //          v2v[v2v.Size() - nnode + 2] = 3;
   //          v2v[v2v.Size() - nnode + 3] = 1;
   //          break;
   //       case 2:
   //          v2v[v2v.Size() - nnode + 0] = 3;
   //          v2v[v2v.Size() - nnode + 1] = 2;
   //          v2v[v2v.Size() - nnode + 2] = 1;
   //          v2v[v2v.Size() - nnode + 3] = 0;
   //          break;
   //       case 3:
   //          v2v[v2v.Size() - nnode + 0] = 1;
   //          v2v[v2v.Size() - nnode + 1] = 3;
   //          v2v[v2v.Size() - nnode + 2] = 0;
   //          v2v[v2v.Size() - nnode + 3] = 2;
   //          break;
   //    }
   //    // renumber elements
   //    for (int i = 0; i < mesh.GetNE(); i++)
   //    {
   //       Element *el = mesh.GetElement(i);
   //       int *v = el->GetVertices();
   //       int nv = el->GetNVertices();
   //       for (int j = 0; j < nv; j++)
   //       {
   //          v[j] = v2v[v[j]];
   //       }
   //    }
   //    // renumber boundary elements
   //    for (int i = 0; i < mesh.GetNBE(); i++)
   //    {
   //       Element *el = mesh.GetBdrElement(i);
   //       int *v = el->GetVertices();
   //       int nv = el->GetNVertices();
   //       for (int j = 0; j < nv; j++)
   //       {
   //          v[j] = v2v[v[j]];
   //       }
   //    }
   //    mesh.RemoveUnusedVertices();
   //    mesh.RemoveInternalBoundaries();
   //
   //    break;
   // }

   // Refine the mesh if desired
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Output the resulting mesh to a file
   {
      ostringstream oss;
      if (el_type_ == Element::QUADRILATERAL)
      {
         oss << "beam-quad";
      }
      else
      {
         oss << "beam-tri";
      }
      oss << "-o" << order_ << "-s" << nx_;
      if (ser_ref_levels > 0)
      {
         oss << "-r" << ser_ref_levels;
      }
      if (per_mesh)
      {
         oss << "-p";
      }
      else if (dg_mesh)
      {
         oss << "-d";
      }
      else
      {
         oss << "-c";
      }

      oss << ".mesh";
      ofstream ofs(oss.str().c_str());
      ofs.precision(8);
      mesh.Print(ofs);
      ofs.close();
   }

   // // Output the resulting mesh to GLVis
   // if (visualization)
   // {
   //    char vishost[] = "localhost";
   //    int  visport   = 19916;
   //    socketstream sol_sock(vishost, visport);
   //    sol_sock.precision(8);
   //    sol_sock << "mesh\n" << mesh << flush;
   // }

   // Clean up and exit
   return 0;
}

// void trans(const Vector &x, Vector &p)
// {
//    double z = x[2];
//    double phi = 0.5 * M_PI * nt_ * z / c_;
//    double cp = cos(phi);
//    double sp = sin(phi);
//
//    p[0] = 0.5 * a_ + (x[0] - 0.5 * a_) * cp - (x[1] - 0.5 * b_) * sp;
//    p[1] = 0.5 * b_ + (x[0] - 0.5 * a_) * sp + (x[1] - 0.5 * b_) * cp;
//    p[2] = z;
// }
