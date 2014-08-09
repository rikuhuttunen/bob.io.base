/**
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the HDF5 (.hdf5) array codec
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include <bob.io.base/File.h>
#include <bob.io.base/HDF5File.h>

#include <boost/make_shared.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

/**
 * Read and write arrays in HDF5 format
 */
class HDF5ArrayFile: public bob::io::base::File {

  public:

    HDF5ArrayFile (const char* filename, bob::io::base::HDF5File::mode_t mode):
      m_file(filename, mode),
      m_filename(filename),
      m_size_arrayset(0),
      m_newfile(true) {

        //tries to update the current descriptors
        std::vector<std::string> paths;
        m_file.paths(paths);

        if (paths.size()) { //file contains data, read it and establish defaults
          m_path = paths[0]; ///< locks on a path name from now on...
          m_newfile = false; ///< blocks re-initialization

          //arrayset reading
          const bob::io::base::HDF5Descriptor& desc_arrayset = m_file.describe(m_path)[0];
          desc_arrayset.type.copy_to(m_type_arrayset);
          m_size_arrayset = desc_arrayset.size;

          //array reading
          const bob::io::base::HDF5Descriptor& desc_array = m_file.describe(m_path)[1];
          desc_array.type.copy_to(m_type_array);

          //if m_type_all has extent == 1 on the first dimension and dimension
          //0 is expandable, collapse that
          if (m_type_array.shape[0] == 1 && desc_arrayset.expandable)
          {
            m_type_array = m_type_arrayset;
          }
        }

        else {
          //default path in case the file is new or has been truncated
          m_path = "/array";
        }

      }

    virtual ~HDF5ArrayFile() { }

    virtual const char* filename() const {
      return m_filename.c_str();
    }

    virtual const BobIoTypeinfo& type_all () const {
      return m_type_array;
    }

    virtual const BobIoTypeinfo& type () const {
      return m_type_arrayset;
    }

    virtual size_t size() const {
      return m_size_arrayset;
    }

    virtual const char* name() const {
      return s_codecname.c_str();
    }

    virtual void read_all(bob::io::base::array::interface& buffer) {

      if(m_newfile) {
        boost::format f("uninitialized HDF5 file at '%s' cannot be read");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      if(!BobIoTypeinfo_IsCompatible(&buffer.type(), &m_type_array)) buffer.set(m_type_array);

      m_file.read_buffer(m_path, 0, buffer.type(), buffer.ptr());
    }

    virtual void read(bob::io::base::array::interface& buffer, size_t index) {

      if(m_newfile) {
        boost::format f("uninitialized HDF5 file at '%s' cannot be read");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      if(!BobIoTypeinfo_IsCompatible(&buffer.type(), &m_type_arrayset)) buffer.set(m_type_arrayset);

      m_file.read_buffer(m_path, index, buffer.type(), buffer.ptr());
    }

    virtual size_t append (const bob::io::base::array::interface& buffer) {

      if (m_newfile) {
        //creates non-compressible, extensible dataset on HDF5 file
        m_newfile = false;
        m_file.create(m_path, buffer.type(), true, 0);
        m_file.describe(m_path)[0].type.copy_to(m_type_arrayset);
        m_file.describe(m_path)[1].type.copy_to(m_type_array);

        //if m_type_all has extent == 1 on the first dimension, collapse that
        if (m_type_array.shape[0] == 1) m_type_array = m_type_arrayset;
      }

      m_file.extend_buffer(m_path, buffer.type(), buffer.ptr());
      ++m_size_arrayset;
      //needs to flush the data to the file
      return m_size_arrayset - 1; ///< index of this object in the file

    }

    virtual void write (const bob::io::base::array::interface& buffer) {

      if (!m_newfile) {
        boost::format f("cannot perform single (array-style) write on file/dataset at '%s' that have already been initialized -- try to use a new file");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      m_newfile = false;
      m_file.create(m_path, buffer.type(), false, 0);

      m_file.describe(m_path)[0].type.copy_to(m_type_arrayset);
      m_file.describe(m_path)[1].type.copy_to(m_type_array);

      //if m_type_all has extent == 1 on the first dimension, collapse that
      if (m_type_array.shape[0] == 1) m_type_array = m_type_arrayset;

      //otherwise, all must be in place...
      m_file.write_buffer(m_path, 0, buffer.type(), buffer.ptr());
    }

  private: //representation

    bob::io::base::HDF5File m_file;
    std::string  m_filename;
    BobIoTypeinfo m_type_array;    ///< type for reading all data at once
    BobIoTypeinfo m_type_arrayset; ///< type for reading data by sub-arrays
    size_t       m_size_arrayset; ///< number of arrays in arrayset mode
    std::string  m_path; ///< default path to use
    bool         m_newfile; ///< path check optimization

    static std::string  s_codecname;

};

std::string HDF5ArrayFile::s_codecname = "bob.hdf5";

/**
 * Registration method: use an unique name. Copy the definition to "plugin.h"
 * and then call it on "main.cpp" to register the codec.
 */
boost::shared_ptr<bob::io::base::File>
  make_hdf5_file (const char* path, char mode) {

  bob::io::base::HDF5File::mode_t h5mode;
  if (mode == 'r') h5mode = bob::io::base::HDF5File::in;
  else if (mode == 'w') h5mode = bob::io::base::HDF5File::trunc;
  else if (mode == 'a') h5mode = bob::io::base::HDF5File::inout;
  else throw std::runtime_error("unsupported file opening mode");

  return boost::make_shared<HDF5ArrayFile>(path, h5mode);

}
