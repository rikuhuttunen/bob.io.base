/**
 * @date Thu 10 May 2012 15:19:24 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Code to read and write CSV files to/from arrays. CSV files are always
 * treated as containing sequences of double precision numbers.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#define BOB_IO_BASE_MODULE
#include <bob.io.base/api.h>
#include <bob.io.base/File.h>

#include <sstream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/tokenizer.hpp>

#include <boost/shared_array.hpp>
#include <boost/algorithm/string.hpp>

typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;

class CSVFile: public bob::io::base::File {

  public: //api

    /**
     * Peeks the file contents for a type. We assume the element type to be
     * always doubles. This method, effectively, only peaks for the total
     * number of lines and the number of columns in the file.
     */
    void peek() {

      std::string line;
      size_t line_number = 0;
      size_t entries = 0;
      std::streampos cur_pos = 0;

      m_file.seekg(0); //< returns to the begin of file and start reading...

      while (std::getline(m_file,line)) {
        ++line_number;
        m_pos.push_back(cur_pos);
        cur_pos = m_file.tellg();
        Tokenizer tok(line);
        size_t size = std::distance(tok.begin(), tok.end());
        if (!entries) entries = size;
        else if (entries != size) {
          boost::format m("line %d at file '%s' contains %d entries instead of %d (expected)");
          m % line_number % m_filename % size % entries;
          throw std::runtime_error(m.str());
        }
      }

      if (!line_number) {
        m_newfile = true;
        m_pos.clear();
        return;
      }

      m_arrayset_type.dtype = NPY_FLOAT64;
      m_arrayset_type.nd = 1;
      m_arrayset_type.shape[0] = entries;
      BobIoTypeinfo_UpdateStrides(&m_arrayset_type);

      m_array_type = m_arrayset_type;
      m_array_type.nd = 2;
      m_array_type.shape[0] = m_pos.size();
      m_array_type.shape[1] = entries;
      BobIoTypeinfo_UpdateStrides(&m_array_type);
    }

    CSVFile(const char* path, char mode):
      m_filename(path),
      m_newfile(false) {

        if (mode == 'r' || (mode == 'a' && boost::filesystem::exists(path))) { //try peeking

          if (mode == 'r')
            m_file.open(m_filename.c_str(), std::ios::in);
          else if (mode == 'a')
            m_file.open(m_filename.c_str(), std::ios::app|std::ios::in|std::ios::out);
          if (!m_file.is_open()) {
            boost::format m("cannot open file '%s' for reading or appending");
            m % path;
            throw std::runtime_error(m.str());
          }

          peek(); ///< peek file properties
        }
        else {
          m_file.open(m_filename.c_str(), std::ios::trunc|std::ios::in|std::ios::out);

          if (!m_file.is_open()) {
            boost::format m("cannot open file '%s' for writing");
            m % path;
            throw std::runtime_error(m.str());
          }

          m_newfile = true;
        }

        //general precision settings, in case output is needed...
        m_file.precision(10);
        m_file.setf(std::ios_base::scientific, std::ios_base::floatfield);

      }

    virtual ~CSVFile() { }

    virtual const char* filename() const {
      return m_filename.c_str();
    }

    virtual const BobIoTypeinfo& type() const {
      return m_arrayset_type;
    }

    virtual const BobIoTypeinfo& type_all() const {
      return m_array_type;
    }

    virtual size_t size() const {
      return m_pos.size();
    }

    virtual const char* name() const {
      return s_codecname.c_str();
    }

    virtual void read_all(bob::io::base::array::interface& buffer) {
      if (m_newfile)
        throw std::runtime_error("uninitialized CSV file cannot be read");

      if (!BobIoTypeinfo_IsCompatible(&buffer.type(), &m_array_type)) buffer.set(m_array_type);

      //read contents
      std::string line;
      if (m_file.eof()) m_file.clear(); ///< clear current "end" state.
      m_file.seekg(0);
      double* p = static_cast<double*>(buffer.ptr());
      while (std::getline(m_file, line)) {
        Tokenizer tok(line);
        for(Tokenizer::iterator k=tok.begin(); k!=tok.end(); ++k) {
          std::istringstream(*k) >> *(p++);
        }
      }
    }

    virtual void read(bob::io::base::array::interface& buffer, size_t index) {

      if (m_newfile)
        throw std::runtime_error("uninitialized CSV file cannot be read");

      if (!BobIoTypeinfo_IsCompatible(&buffer.type(), &m_arrayset_type))
        buffer.set(m_arrayset_type);

      if (index >= m_pos.size()) {
        boost::format m("cannot array at position %d -- there is only %d entries at file '%s'");
        m % index % m_pos.size() % m_filename;
        throw std::runtime_error(m.str());
      }

      //reads a specific line from the file.
      std::string line;
      if (m_file.eof()) m_file.clear(); ///< clear current "end" state.
      m_file.seekg(m_pos[index]);
      if (!std::getline(m_file, line)) {
        boost::format m("could not seek to line %u (offset %u) while reading file '%s'");
        m % index % m_pos[index] % m_filename;
        throw std::runtime_error(m.str());
      }
      Tokenizer tok(line);
      double* p = static_cast<double*>(buffer.ptr());
      for(Tokenizer::iterator k=tok.begin(); k!=tok.end(); ++k) {
        std::istringstream(*k) >> *(p++);
      }

    }

    virtual size_t append (const bob::io::base::array::interface& buffer) {

      const BobIoTypeinfo& type = buffer.type();

      if (m_newfile) {
        if (type.nd != 1 || type.dtype != NPY_FLOAT64) {
          boost::format m("cannot append %s to file '%s' - CSV files only accept 1D double precision float arrays");
          m % BobIoTypeinfo_Str(&type) % m_filename;
          throw std::runtime_error(m.str());
        }
        m_pos.clear();
        m_arrayset_type = m_array_type = type;
        m_array_type.shape[1] = m_arrayset_type.shape[0];
        m_newfile = false;
      }

      else {

        //check compatibility
        if (!BobIoTypeinfo_IsCompatible(&m_arrayset_type, &buffer.type())) {
          boost::format m("CSV file '%s' only accepts arrays of type %s");
          m % m_filename % BobIoTypeinfo_Str(&m_arrayset_type);
          throw std::runtime_error(m.str());
        }

      }

      const double* p = static_cast<const double*>(buffer.ptr());
      if (m_pos.size()) m_file << std::endl; ///< adds a new line
      m_pos.push_back(m_file.tellp()); ///< register start of line
      for (size_t k=1; k<type.shape[0]; ++k) m_file << *(p++) << ",";
      m_file << *(p++);
      m_array_type.shape[0] = m_pos.size();
      BobIoTypeinfo_UpdateStrides(&m_array_type);
      return (m_pos.size()-1);

    }

    virtual void write (const bob::io::base::array::interface& buffer) {

      const BobIoTypeinfo& type = buffer.type();

      if (m_newfile) {
        if (type.nd != 2 || type.dtype != NPY_FLOAT64) {
          boost::format m("cannot write %s to file '%s' - CSV files only accept a single 2D double precision float array as input");
          m % BobIoTypeinfo_Str(&type) % m_filename;
          throw std::runtime_error(m.str());
        }
        const double* p = static_cast<const double*>(buffer.ptr());
        for (size_t l=1; l<type.shape[0]; ++l) {
          m_pos.push_back(m_file.tellp());
          for (size_t k=1; k<type.shape[1]; ++k) m_file << *(p++) << ",";
          m_file << *(p++) << std::endl;
        }
        for (size_t k=1; k<type.shape[1]; ++k) m_file << *(p++) << ",";
        m_file << *(p++);
        m_arrayset_type = type;
        m_arrayset_type.nd = 1;
        m_arrayset_type.shape[0] = type.shape[1];
        BobIoTypeinfo_UpdateStrides(&m_arrayset_type);
        m_array_type = type;
        m_newfile = false;
        return;
      }

      //TODO
      throw std::runtime_error("Writing a 2D array to a CSV file that already contains entries is not implemented at the moment");

    }

  private: //representation
    std::fstream m_file;
    std::string m_filename;
    bool m_newfile;
    BobIoTypeinfo m_array_type;
    BobIoTypeinfo m_arrayset_type;
    std::vector<std::streampos> m_pos; ///< dictionary of line starts

    static std::string s_codecname;

};

std::string CSVFile::s_codecname = "bob.csv";

/**
 * Registration method: use an unique name. Copy the definition to "plugin.h"
 * and then call it on "main.cpp" to register the codec.
 */
boost::shared_ptr<bob::io::base::File>
  make_csv_file (const char* path, char mode) {
  return boost::make_shared<CSVFile>(path, mode);
}
