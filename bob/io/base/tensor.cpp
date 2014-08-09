/**
 * @date Wed Oct 26 17:11:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the TensorArrayCodec type
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include <bob.io.base/File.h>

#include "cpp/TensorFile.h"

class TensorArrayFile: public bob::io::base::File {

  public: //api

    TensorArrayFile(const char* path, bob::io::base::TensorFile::openmode mode):
      m_file(path, mode),
      m_filename(path) {
        if (m_file.size()) m_file.peek(m_type);
      }

    virtual ~TensorArrayFile() { }

    virtual const char* filename() const {
      return m_filename.c_str();
    }

    virtual const BobIoTypeinfo& type_all () const {
      return m_type;
    }

    virtual const BobIoTypeinfo& type () const {
      return m_type;
    }

    virtual size_t size() const {
      return m_file.size();
    }

    virtual const char* name() const {
      return s_codecname.c_str();
    }

    virtual void read_all(bob::io::base::array::interface& buffer) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(0, buffer);

    }

    virtual void read(bob::io::base::array::interface& buffer, size_t index) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(index, buffer);

    }

    virtual size_t append (const bob::io::base::array::interface& buffer) {

      m_file.write(buffer);

      if (size() == 1) m_file.peek(m_type);

      return size() - 1;

    }

    virtual void write (const bob::io::base::array::interface& buffer) {

      //we don't have a special way to treat write()'s like in HDF5.
      append(buffer);

    }

  private: //representation

    bob::io::base::TensorFile m_file;
    BobIoTypeinfo m_type;
    std::string m_filename;

    static std::string s_codecname;

};

std::string TensorArrayFile::s_codecname = "bob.tensor";

/**
 * Registration method: use an unique name. Copy the definition to "plugin.h"
 * and then call it on "main.cpp" to register the codec.
 */
boost::shared_ptr<bob::io::base::File>
  make_tensor_file (const char* path, char mode) {

  bob::io::base::TensorFile::openmode _mode;
  if (mode == 'r') _mode = bob::io::base::TensorFile::in;
  else if (mode == 'w') _mode = bob::io::base::TensorFile::out;
  else if (mode == 'a') _mode = bob::io::base::TensorFile::append;
  else throw std::runtime_error("unsupported tensor file opening mode");

  return boost::make_shared<TensorArrayFile>(path, _mode);

}
