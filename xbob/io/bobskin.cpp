/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 07:57:57 2013
 *
 * @brief Implementation of our bobskin class
 */

#include <numpy/arrayobject.h>
#include <stdexcept>

bobskin::bobskin(PyObject* array, bob::core::array::ElementType& eltype) {

  if (!PyArray_CheckExact(array)) {
    PyErr_SetString(PyExc_TypeError, "input object to bobskin constructor is not a numpy.ndarray");
    throw std::runtime_error();
  }

  m_type.set(eltype, PyArray_NDIM(array), PyArray_DIMS(array),
      PyArray_STRIDES(array));

  m_ptr = PyArray_DATA(array);

}

bobskin::~bobskin() { }

void bobskin::set(const interface&) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const interface&) is not implemented - DEBUG ME!");
  throw std::runtime_error();
}

void bobskin::set(boost::shared_ptr<interface> other);
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (boost::shared_ptr<interface>) is not implemented - DEBUG ME!");
  throw std::runtime_error();
}

void bobskin::set (const bob::core::array::typeinfo& req) {
  PyErr_SetString(PyExc_NotImplementedError, "setting C++ bobskin with (const typeinfo&) implemented - DEBUG ME!");
  throw std::runtime_error();
}

boost::shared_ptr<void> bobskin::owner() {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring non-const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error();
}

boost::shared_ptr<const void> bobskin::owner() const {
  PyErr_SetString(PyExc_NotImplementedError, "acquiring const owner from C++ bobskin is not implemented - DEBUG ME!");
  throw std::runtime_error();
}
