/**
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief The array API describes a non-specific way to handle N dimensional
 * array data.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_BASE_ARRAY_INTERFACE_H
#define BOB_IO_BASE_ARRAY_INTERFACE_H

#include <bob.blitz/cppapi.h>
#include <bob.io.base/api.h>

#include <stdexcept>
#include <string>

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

/* MinGW flags */
#ifdef _WIN32
#undef interface
#endif

/**
 * @brief Array submodule API of the I/O module
 */
namespace bob { namespace io { namespace base { namespace array {

  /**
   * @brief The interface manager introduces a concept for managing the
   * interfaces that can be handled as C-style arrays. It encapsulates methods
   * to store and delete the buffer contents in a safe way.
   *
   * The interface is an entity that either stores a copy of its own data or
   * refers to data belonging to another interface.
   */
  class interface {

    public: //api

      /**
       * @brief By default, the interface is never freed. You must override
       * this method to do something special for your class type.
       */
      virtual ~interface() { }

      /**
       * @brief Copies the data from another interface.
       */
      virtual void set(const interface& other) =0;

      /**
       * @brief Refers to the data of another interface.
       */
      virtual void set(boost::shared_ptr<interface> other) =0;

      /**
       * @brief Re-allocates this interface taking into consideration new
       * requirements. The internal memory should be considered uninitialized.
       */
      virtual void set (const BobIoTypeinfo& req) =0;

      /**
       * @brief Type information for this interface.
       */
      virtual const BobIoTypeinfo& type() const =0;

      /**
       * @brief Borrows a reference from the underlying memory. This means
       * this object continues to be responsible for deleting the memory and
       * you should make sure that it outlives the usage of the returned
       * pointer.
       */
      virtual void* ptr() =0;
      virtual const void* ptr() const =0;

      /**
       * @brief Returns a representation of the internal cache using shared
       * pointers.
       */
      virtual boost::shared_ptr<void> owner() =0;
      virtual boost::shared_ptr<const void> owner() const =0;

  };

}}}}

#endif /* BOB_IO_BASE_ARRAY_INTERFACE_H */
