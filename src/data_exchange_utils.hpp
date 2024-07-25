#include "tps.hpp"

#ifndef DATA_EXCHANGE_UTILS
#define DATA_EXCHANGE_UTILS

namespace TPS {

class CPUDataRead {
 public:
  CPUDataRead(const mfem::Vector &v) : data_(v.HostRead()), size_(v.Size()), stride_(1) {}
  double *data() const { return const_cast<double *>(data_); }
  size_t size() const { return size_; }
  size_t stride() const { return stride_; }

 private:
  const double *data_;
  size_t size_;
  size_t stride_;
};

class CPUData {
 public:
  CPUData(mfem::Vector &v, bool rw) : data_(rw ? v.HostReadWrite() : v.HostWrite()), size_(v.Size()), stride_(1) {}
  double *data() { return data_; }
  size_t size() const { return size_; }
  size_t stride() const { return stride_; }

 private:
  double *data_;
  size_t size_;
  size_t stride_;
};
}

#endif