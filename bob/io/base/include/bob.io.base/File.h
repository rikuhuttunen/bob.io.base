/**
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_BASE_FILE_H
#define BOB_IO_BASE_FILE_H

#include <bob.io.base/blitz_array.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace io { namespace base {

  /**
   * @brief Files deal with reading and writing multiple (homogeneous) array
   * data to and from files.
   */
  class File {

    public: //abstract API

      virtual ~File() {};

      /**
       * The filename this array codec current points to
       */
      virtual const char* filename() const =0;

      /**
       * The type information of data within this file, if it is supposed to be
       * read as as a sequence of arrays
       */
      virtual const BobIoTypeinfo& type() const =0;

      /**
       * The type information of data within this file, if it is supposed to be
       * read as a single array.
       */
      virtual const BobIoTypeinfo& type_all() const =0;

      /**
       * The number of arrays available in this file, if it is supposed to be
       * read as a sequence of arrays.
       */
      virtual size_t size() const =0;

      /**
       * Returns the name of the codec, for compatibility reasons.
       */
      virtual const char* name() const =0;

      /**
       * Loads the data of the array into memory. If an index is specified
       * loads the specific array data from the file, otherwise, loads the data
       * at position 0.
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocate enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void read(bob::io::base::array::interface& buffer, size_t index) =0;

      /**
       * Loads all the data available at the file into a single in-memory
       * array.
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocate enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void read_all(bob::io::base::array::interface& buffer) =0;

      /**
       * Appends the given buffer into a file. If the file does not exist,
       * create a new file, else, makes sure that the inserted array respects
       * the previously set file structure.
       *
       * Returns the current position of the newly written array.
       */
      virtual size_t append (const bob::io::base::array::interface& buffer) =0;

      /**
       * Writes the data from the given buffer into the file and act like it is
       * the only piece of data that will ever be written to such a file. Not
       * more data appending may happen after a call to this method.
       */
      virtual void write (const bob::io::base::array::interface& buffer) =0;

    public: //blitz::Array specific API

      /**
       * This method returns a copy of the array in the file with the element
       * type you wish (just have to get the number of dimensions right!).
       */
      template <typename T, int N> blitz::Array<T,N> cast(size_t index) {
        bob::io::base::array::blitz_array tmp(type());
        read(tmp, index);
        return tmp.cast<T,N>();
      }

      /**
       * This method returns a copy of the array in the file with the element
       * type you wish (just have to get the number of dimensions right!).
       *
       * This variant loads all data available into the file in a single array.
       */
      template <typename T, int N> blitz::Array<T,N> cast_all() {
        bob::io::base::array::blitz_array tmp(type_all());
        read_all(tmp);
        return tmp.cast<T,N>();
      }

      template <typename T, int N> void read(blitz::Array<T,N>& io,
          size_t index) {
        bob::io::base::array::blitz_array use_this(io);
        use_this.set(type());
        read(use_this, index);
        io.reference(use_this.get<T,N>());
      }

      template <typename T, int N> blitz::Array<T,N> read(size_t index) {
        bob::io::base::array::blitz_array tmp(type());
        read(tmp, index);
        return tmp.get<T,N>();
      }

      template <typename T, int N> void read_all(blitz::Array<T,N>& io) {
        bob::io::base::array::blitz_array use_this(io);
        use_this.set(type_all());
        read_all(use_this);
        io.reference(use_this.get<T,N>());
      }

      template <typename T, int N> blitz::Array<T,N> read_all() {
        bob::io::base::array::blitz_array tmp(type_all());
        read_all(tmp);
        return tmp.get<T,N>();
      }

      template <typename T, int N> size_t append(const blitz::Array<T,N>& in) {
        bob::io::base::array::blitz_array use_this(in);
        return append(use_this);
      }

      template <typename T, int N> void write (const blitz::Array<T,N>& in) {
        bob::io::base::array::blitz_array use_this(in);
        write(use_this);
      }

  };

}}}

#endif /* BOB_IO_BASE_FILE_H */
