/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 07:57:57 2013
 *
 * @brief Implementation of our bobskin class
 */

#include "bobskin.h"
#include <stdexcept>

bobskin::bobskin(PyObject* array, int dtype) {

  if (!PyArray_CheckExact(array)) {
    PyErr_SetString(PyExc_TypeError, "input object to bobskin constructor is not (exactly) a numpy.ndarray");
    throw std::runtime_error("error is already set");
  }

  if (!BobIoTypeinfo_SignedSetWithStrides(&m_type, dtype,
        PyArray_NDIM((PyArrayObject*)array),
        PyArray_DIMS((PyArrayObject*)array),
        PyArray_STRIDES((PyArrayObject*)array))) {
    throw std::runtime_error("error is already set");
  }

  m_ptr = PyArray_DATA((PyArrayObject*)array);

}

bobskin::bobskin(PyArrayObject* array, int dtype) {

  if (!BobIoTypeinfo_SignedSetWithStrides(&m_type, dtype,
        PyArray_NDIM((PyArrayObject*)array),
        PyArray_DIMS((PyArrayObject*)array),
        PyArray_STRIDES((PyArrayObject*)array))) {
    throw std::runtime_error("error is already set");
  }

  m_ptr = PyArray_DATA((PyArrayObject*)array);

}

bobskin::bobskin(PyBlitzArrayObject* array) {
  if (!BobIoTypeinfo_SignedSetWithStrides(&m_type, array->type_num,
        array->ndim, array->shape, array->stride)) {
    throw std::runtime_error("error is already set");
  }
  m_ptr = array->data;
}

bobskin::~bobskin() { }

void bobskin::set(const interface&) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const interface&) is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

void bobskin::set(boost::shared_ptr<interface>) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (boost::shared_ptr<interface>) is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

void bobskin::set (const BobIoTypeinfo&) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const type-information&) implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

boost::shared_ptr<void> bobskin::owner() {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring non-const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}

boost::shared_ptr<const void> bobskin::owner() const {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error("error is already set");
}
