/**
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of non-templated methods of the blitz
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <stdexcept>

#include <bob.io.base/blitz_array.h>

bob::io::base::array::blitz_array::blitz_array(boost::shared_ptr<blitz_array> other) {
  set(other);
}

bob::io::base::array::blitz_array::blitz_array(const blitz_array& other) {
  set(other);
}

bob::io::base::array::blitz_array::blitz_array(boost::shared_ptr<interface> other) {
  set(other);
}

bob::io::base::array::blitz_array::blitz_array(const interface& other) {
  set(other);
}

bob::io::base::array::blitz_array::blitz_array(const BobIoTypeinfo& info) {
  set(info);
}

bob::io::base::array::blitz_array::blitz_array(void* data, const BobIoTypeinfo& info):
  m_type(),
  m_ptr(data),
  m_is_blitz(false) {
  if (!BobIoTypeinfo_Copy(&m_type, &info)) {
    throw std::runtime_error("error already set");
  }
}

bob::io::base::array::blitz_array::~blitz_array() {
}

void bob::io::base::array::blitz_array::set(boost::shared_ptr<blitz_array> other) {
  if (!BobIoTypeinfo_Copy(&m_type, &other->m_type)) {
    throw std::runtime_error("error already set");
  }
  m_ptr = other->m_ptr;
  m_is_blitz = other->m_is_blitz;
  m_data = other->m_data;
}

void bob::io::base::array::blitz_array::set(const interface& other) {
  set(other.type());
  memcpy(m_ptr, other.ptr(), BobIoTypeinfo_BufferSize(&m_type));
}

void bob::io::base::array::blitz_array::set(boost::shared_ptr<interface> other) {
  if (!BobIoTypeinfo_Copy(&m_type, &other->type())) {
    throw std::runtime_error("error already set");
  }
  m_ptr = other->ptr();
  m_is_blitz = false;
  m_data = other;
}

template <typename T>
static boost::shared_ptr<void> make_array(size_t nd, const size_t* shape,
    void*& ptr) {
  switch(nd) {
    case 1:
      {
        blitz::TinyVector<int,1> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval =
          boost::make_shared<blitz::Array<T,1> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,1> >(retval)->data());
        return retval;
      }
    case 2:
      {
        blitz::TinyVector<int,2> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval =
          boost::make_shared<blitz::Array<T,2> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,2> >(retval)->data());
        return retval;
      }
    case 3:
      {
        blitz::TinyVector<int,3> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval =
          boost::make_shared<blitz::Array<T,3> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,3> >(retval)->data());
        return retval;
      }
    case 4:
      {
        blitz::TinyVector<int,4> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval =
          boost::make_shared<blitz::Array<T,4> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,4> >(retval)->data());
        return retval;
      }
    default:
      break;
  }
  throw std::runtime_error("unsupported number of dimensions -- debug me");
}

void bob::io::base::array::blitz_array::set (const BobIoTypeinfo& req) {
  if (BobIoTypeinfo_IsCompatible(&m_type, &req)) return; ///< double-check requirement first!

  //have to go through reallocation
  if (!BobIoTypeinfo_Copy(&m_type, &req)) {
    throw std::runtime_error("error already set");
  }

  m_is_blitz = true;

  switch (m_type.dtype) {
    case NPY_BOOL:
      m_data = make_array<bool>(req.nd, req.shape, m_ptr);
      return;
    case NPY_INT8:
      m_data = make_array<int8_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_INT16:
      m_data = make_array<int16_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_INT32:
      m_data = make_array<int32_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_INT64:
      m_data = make_array<int64_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_UINT8:
      m_data = make_array<uint8_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_UINT16:
      m_data = make_array<uint16_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_UINT32:
      m_data = make_array<uint32_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_UINT64:
      m_data = make_array<uint64_t>(req.nd, req.shape, m_ptr);
      return;
    case NPY_FLOAT32:
      m_data = make_array<float>(req.nd, req.shape, m_ptr);
      return;
    case NPY_FLOAT64:
      m_data = make_array<double>(req.nd, req.shape, m_ptr);
      return;
#   ifdef NPY_FLOAT128
    case NPY_FLOAT128:
      m_data = make_array<long double>(req.nd, req.shape, m_ptr);
      return;
#   endif
    case NPY_COMPLEX64:
      m_data = make_array<std::complex<float> >(req.nd, req.shape, m_ptr);
      return;
    case NPY_COMPLEX128:
      m_data = make_array<std::complex<double> >(req.nd, req.shape, m_ptr);
      return;
#   ifdef NPY_COMPLEX256
    case NPY_COMPLEX256:
      m_data = make_array<std::complex<long double> >(req.nd, req.shape, m_ptr);
      return;
#   endif
    default:
      break;
  }

  //if we get to this point, there is nothing much we can do...
  throw std::runtime_error("invalid data type on blitz array reset -- debug me");
}
