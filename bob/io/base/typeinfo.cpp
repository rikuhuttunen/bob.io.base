/**
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Some buffer stuff
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.blitz/capi.h>
#include <bob.io.base/api.h>
#include <boost/format.hpp>

void BobIoTypeinfo_Init(BobIoTypeinfo* self) {
  BobIoTypeinfo_Reset(self);
}

int BobIoTypeinfo_Copy (BobIoTypeinfo* self, const BobIoTypeinfo* other) {

  self->dtype = other->dtype;

  if (other->nd > (BOB_BLITZ_MAXDIMS+1)) {
    PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", other->nd, BOB_BLITZ_MAXDIMS);
    return 0;
  }

  self->nd = other->nd;
  for (size_t k=0; k<self->nd; ++k) self->shape[k] = other->shape[k];

  return BobIoTypeinfo_UpdateStrides(self);

}

int BobIoTypeinfo_Set (BobIoTypeinfo* self, int dtype, size_t nd,
    const size_t* shape) {

  self->dtype = dtype;

  if (nd > (BOB_BLITZ_MAXDIMS+1)) {
    PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", nd, BOB_BLITZ_MAXDIMS);
    return 0;
  }

  self->nd = nd;
  for (size_t k=0; k<nd; ++k) self->shape[k] = shape[k];

  return BobIoTypeinfo_UpdateStrides(self);

}

int BobIoTypeinfo_SignedSet (BobIoTypeinfo* self, int dtype, Py_ssize_t nd,
    const Py_ssize_t* shape) {

  self->dtype = dtype;

  if (nd > (BOB_BLITZ_MAXDIMS+1)) {
    PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", nd, BOB_BLITZ_MAXDIMS);
    return 0;
  }

  self->nd = nd;
  for (Py_ssize_t k=0; k<nd; ++k) self->shape[k] = shape[k];

  return BobIoTypeinfo_UpdateStrides(self);

}

int BobIoTypeinfo_SetWithStrides (BobIoTypeinfo* self, int dtype,
    Py_ssize_t nd, const Py_ssize_t* shape, const Py_ssize_t* stride) {
  self->dtype = dtype;

  if (nd > (BOB_BLITZ_MAXDIMS+1)) {
    PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", nd, BOB_BLITZ_MAXDIMS);
    return 0;
  }

  self->nd = nd;
  for (Py_ssize_t k=0; k<nd; ++k) {
    self->shape[k] = shape[k];
    self->stride[k] = stride[k];
  }

  return 1;

}

int BobIoTypeinfo_SignedSetWithStrides (BobIoTypeinfo* self, int dtype,
    Py_ssize_t nd, const Py_ssize_t* shape, const Py_ssize_t* stride) {
  self->dtype = dtype;

  if (nd > (BOB_BLITZ_MAXDIMS+1)) {
    PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", nd, BOB_BLITZ_MAXDIMS);
    return 0;
  }

  self->nd = nd;
  for (Py_ssize_t k=0; k<nd; ++k) {
    self->shape[k] = shape[k];
    self->stride[k] = stride[k];
  }

  return 1;

}

void BobIoTypeinfo_Reset(BobIoTypeinfo* self) {

  self->dtype = NPY_NOTYPE;
  self->nd = 0;

}

bool BobIoTypeinfo_IsValid(const BobIoTypeinfo* self) {

  return (self->dtype != NPY_NOTYPE) && (self->nd > 0) && (self->nd <= (BOB_BLITZ_MAXDIMS+1)) && BobIoTypeinfo_HasValidShape(self);

}

int BobIoTypeinfo_UpdateStrides(BobIoTypeinfo* self) {

  auto* stride = self->stride;
  auto* shape = self->shape;

  switch (self->nd) {

    case 0:
      return 1;

    case 1:
      stride[0] = 1;
      return 1;

    case 2:
      stride[1] = 1;
      stride[0] = shape[1];
      return 1;

    case 3:
      stride[2] = 1;
      stride[1] = shape[2];
      stride[0] = shape[1]*shape[2];
      return 1;

    case 4:
      stride[3] = 1;
      stride[2] = shape[3];
      stride[1] = shape[2]*shape[3];
      stride[0] = shape[1]*shape[2]*shape[3];
      return 1;

    case 5:
      stride[4] = 1;
      stride[3] = shape[4];
      stride[2] = shape[3]*shape[4];
      stride[1] = shape[2]*shape[3]*shape[4];
      stride[0] = shape[1]*shape[2]*shape[3]*shape[4];
      return 1;

    default:
      break;

  }

  PyErr_Format(PyExc_RuntimeError, "unsupported number of dimensions (%" PY_FORMAT_SIZE_T "d) while the maximum is %d", self->nd, BOB_BLITZ_MAXDIMS);
  return 0;

}

size_t BobIoTypeinfo_Size(const BobIoTypeinfo* self) {

  size_t retval = 1;
  for (size_t k=0; k<self->nd; ++k) retval *= self->shape[k];
  return retval;

}

size_t BobIoTypeinfo_BufferSize(const BobIoTypeinfo* self) {

  return BobIoTypeinfo_Size(self) * PyBlitzArray_TypenumSize(self->dtype);

}

static bool same_shape(size_t nd, const size_t* s1, const size_t* s2) {

  for (size_t k=0; k<nd; ++k) if (s1[k] != s2[k]) return false;
  return true;

}

bool BobIoTypeinfo_IsCompatible(const BobIoTypeinfo* self,
    const BobIoTypeinfo* other) {

  return (self->dtype == other->dtype) && (self->nd == other->nd) && same_shape(self->nd, self->shape, other->shape);

}

std::string BobIoTypeinfo_Str(const BobIoTypeinfo* self) {

  boost::format s("dtype: %s (%d); shape: [%s]; size: %d bytes");
  size_t sz = 0;
  size_t buf_sz = 0;
  if (self->dtype != NPY_NOTYPE) {
    //otherwise it throws
    sz = PyBlitzArray_TypenumSize(self->dtype);
    buf_sz = BobIoTypeinfo_BufferSize(self);
  }
  s % PyBlitzArray_TypenumAsString(self->dtype) % sz;

  auto* shape = self->shape;

  switch (self->nd) {

    case 0:
      s % "";
      break;

    case 1:
      s % (boost::format("%d") % shape[0]).str();
      break;

    case 2:
      s % (boost::format("%d,%d") % shape[0] % shape[1]).str();
      break;

    case 3:
      s % (boost::format("%d,%d,%d") % shape[0] % shape[1] % shape[2]).str();
      break;

    case 4:
      s % (boost::format("%d,%d,%d,%d") % shape[0] % shape[1] % shape[2] % shape[3]).str();
      break;

    default:
      s % ">4 dimensions?";
      break;

  }

  s % buf_sz;
  return s.str();

}

void BobIoTypeinfo_ResetShape (BobIoTypeinfo* self) {
  self->shape[0] = 0;
}

bool BobIoTypeinfo_HasValidShape(const BobIoTypeinfo* self) {
  return self->shape[0] != 0;
}
