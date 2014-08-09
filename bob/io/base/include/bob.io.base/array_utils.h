/**
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Utilities for converting data to-from blitz::Arrays and other
 * goodies.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_BASE_ARRAY_UTILS_H
#define BOB_IO_BASE_ARRAY_UTILS_H

#include <blitz/array.h>
#include <stdint.h>
#include <stdexcept>
#include <boost/format.hpp>

#include <bob.core/cast.h>
#include <bob.io.base/array.h>

namespace bob { namespace io { namespace base { namespace array {

  /**
   * @brief Fills in shape and stride starting from a type information object
   */
  template <int N> void set_shape_and_stride(const BobIoTypeinfo& info,
      blitz::TinyVector<int,N>& shape, blitz::TinyVector<int,N>& stride) {
    for (int k=0; k<N; ++k) {
      shape[k] = info.shape[k];
      stride[k] = info.stride[k];
    }
  }


  /**
   * @brief Takes a data pointer and assumes it is a C-style array for the
   * defined type. Creates a wrapper as a blitz::Array<T,N> with the same
   * number of dimensions and type. Notice that the blitz::Array<> created
   * will have its memory tied to the passed buffer. In other words you have
   * to make sure that the buffer outlives the returned blitz::Array<>.
   */
  template <typename T, int N>
    blitz::Array<T,N> wrap(const interface& buf) {

      const BobIoTypeinfo& type = buf.type();

      if (!buf.ptr()) throw std::runtime_error("empty buffer");

      if (type.dtype != PyBlitzArrayCxx_CToTypenum<T>()) {
        boost::format m("cannot efficiently retrieve blitz::Array<%s,%d> from buffer of type '%s'");
        m % PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()) % N % BobIoTypeinfo_Str(&type);
        throw std::runtime_error(m.str());
      }

      if (type.nd != N) {
        boost::format m("cannot retrieve blitz::Array<%s,%d> from buffer of type '%s'");
        m % PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()) % N % BobIoTypeinfo_Str(&type);
        throw std::runtime_error(m.str());
      }

      blitz::TinyVector<int,N> shape;
      blitz::TinyVector<int,N> stride;
      set_shape_and_stride(type, shape, stride);

      return blitz::Array<T,N>((T*)buf.ptr(),
          shape, stride, blitz::neverDeleteData);
    }


  /**
   * @brief Takes a data pointer and assumes it is a C-style array for the
   * defined type. Creates a copy as a blitz::Array<T,N> with the same number
   * of dimensions, but with a type as specified by you. If the type does not
   * match the type of the original C-style array, a cast will happen.
   *
   * If a certain type cast is not supported. An appropriate exception will
   * be raised.
   */
  template <typename T, int N>
    blitz::Array<T,N> cast(const interface& buf) {

      const BobIoTypeinfo& type = buf.type();

      if (type.nd != N) {
        boost::format m("cannot cast blitz::Array<%s,%d> from buffer of type '%s'");
        m % PyBlitzArray_TypenumAsString(PyBlitzArrayCxx_CToTypenum<T>()) % N % BobIoTypeinfo_Str(&type);
        throw std::runtime_error(m.str());
      }

      switch (type.dtype) {
        case NPY_BOOL:
          return bob::core::array::cast<T>(wrap<bool,N>(buf));
        case NPY_INT8:
          return bob::core::array::cast<T>(wrap<int8_t,N>(buf));
        case NPY_INT16:
          return bob::core::array::cast<T>(wrap<int16_t,N>(buf));
        case NPY_INT32:
          return bob::core::array::cast<T>(wrap<int32_t,N>(buf));
        case NPY_INT64:
          return bob::core::array::cast<T>(wrap<int64_t,N>(buf));
        case NPY_UINT8:
          return bob::core::array::cast<T>(wrap<uint8_t,N>(buf));
        case NPY_UINT16:
          return bob::core::array::cast<T>(wrap<uint16_t,N>(buf));
        case NPY_UINT32:
          return bob::core::array::cast<T>(wrap<uint32_t,N>(buf));
        case NPY_UINT64:
          return bob::core::array::cast<T>(wrap<uint64_t,N>(buf));
        case NPY_FLOAT32:
          return bob::core::array::cast<T>(wrap<float,N>(buf));
        case NPY_FLOAT64:
          return bob::core::array::cast<T>(wrap<double,N>(buf));
#       ifdef NPY_FLOAT128
        case NPY_FLOAT128:
          return bob::core::array::cast<T>(wrap<long double,N>(buf));
#       endif
        case NPY_COMPLEX64:
          return bob::core::array::cast<T>(wrap<std::complex<float>,N>(buf));
        case NPY_COMPLEX128:
          return bob::core::array::cast<T>(wrap<std::complex<double>,N>(buf));
#       ifdef NPY_COMPLEX256
        case NPY_COMPLEX256:
          return bob::core::array::cast<T>(wrap<std::complex<long double>,N>(buf));
#       endif
        default:
          break;
      }

      //if we get to this point, there is nothing much we can do...
      throw std::runtime_error("invalid type on blitz buffer array casting -- debug me");

    }

}}}}

#endif /* BOB_IO_BASE_ARRAY_UTILS_H */
