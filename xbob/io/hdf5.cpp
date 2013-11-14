/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 12 Nov 18:19:22 2013
 *
 * @brief Bindings to bob::io::HDF5File
 */

#define XBOB_IO_MODULE
#include <xbob.io/api.h>

#include <boost/make_shared.hpp>
#include <numpy/arrayobject.h>
#include <blitz.array/cppapi.h>
#include <stdexcept>
#include <bobskin.h>

#define HDF5FILE_NAME HDF5File
PyDoc_STRVAR(s_hdf5file_str, BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(HDF5FILE_NAME));

PyDoc_STRVAR(s_hdf5file_doc,
"HDF5File(filename, [mode='r']) -> new bob::io::HDF5File\n\
\n\
Reads and writes data to HDF5 files.\n\
\n\
Constructor parameters:\n\
\n\
filename\n\
  [str] The file path to the file you want to read from/write to\n\
\n\
mode\n\
  [str, optional] The opening mode: Use ``'r'`` for read-only,\n\
  ``'a'`` for read/write/append, ``'w'`` for read/write/truncate\n\
  or ``'x'`` for (read/write/exclusive). This flag defaults to\n\
  ``'r'``.\n\
\n\
HDF5 stands for Hierarchical Data Format version 5. It is a\n\
flexible, binary file format that allows one to store and read\n\
data efficiently into files. It is a cross-platform,\n\
cross-architecture format.\n\
\n\
Objects of this class allows users to read and write data from\n\
and to files in HDF5 format. For an introduction to HDF5, visit\n\
the `HDF5 Website<http://www.hdfgroup.org/HDF5>`_.\n\
\n\
");

int PyBobIoHDF5File_Check(PyObject* o) {
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIoHDF5File_Type));
}

int PyBobIoHDF5File_Converter(PyObject* o, PyBobIoHDF5FileObject** a) {
  if (!PyBobIoHDF5File_Check(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<PyBobIoHDF5FileObject*>(o);
  return 1;
}

/* How to create a new PyBobIoHDF5FileObject */
static PyObject* PyBobIoHDF5File_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoHDF5FileObject* self = (PyBobIoHDF5FileObject*)type->tp_alloc(type, 0);

  self->f.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoHDF5File_Delete (PyBobIoHDF5FileObject* o) {

  o->f.reset();
  o->ob_type->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoHDF5File_Init(PyBobIoHDF5FileObject* self, 
    PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"filename", "mode", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* filename = 0;
  char mode = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|c", kwlist, &filename, &mode))
    return -1;

  if (mode != 'r' && mode != 'w' && mode != 'a' && mode != 'x') {
    PyErr_Format(PyExc_ValueError, "file open mode string should have 1 element and be either 'r' (read), 'w' (write), 'a' (append), 'x' (exclusive)");
    return -1;
  }

  try {
    self->f.reset(new bob::io::HDF5File(filename, mode));
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "cannot open file `%s' with mode `%c': %s", filename, mode, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot open file `%s' with mode `%c': unknown exception caught", filename, mode);
    return -1;
  }

  return 0; ///< SUCCESS
}

static PyObject* PyBobIoHDF5File_Repr(PyBobIoHDF5FileObject* self) {
  return
# if PY_VERSION_HEX >= 0x03000000
  PyUnicode_FromFormat
# else
  PyString_FromFormat
# endif
  ("%s(filename='%s')", s_hdf5file_str, self->f->filename().c_str());
}

static PyObject* PyBobIoHDF5File_ChangeDirectory(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  try {
    self->f->cd(path);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "cannot change directory to `%s' in file `%s': %s", path, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while changing directory to `%s' in HDF5 file `%s'", path, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_cd_str, "cd");
PyDoc_STRVAR(s_cd_doc,
"x.cd(path) -> None\n\
\n\
Changes the current prefix path.\n\
\n\
Parameters:\n\
\n\
path\n\
  [str] The path to change directories to\n\
\n\
When this object is started, the prefix path is empty, which\n\
means all following paths to data objects should be given using\n\
the full path. If you set this to a different value, it will be\n\
used as a prefix to any subsequent operation until you reset\n\
it. If path starts with ``'/'``, it is treated as an absolute\n\
path. ``'..'`` and ``'.'`` are supported. This object should\n\
be an :py:class:`str` object. If the value is relative, it is\n\
added to the current path. If it is absolute, it causes the\n\
prefix to be reset. Note all operations taking a relative path,\n\
following a ``cd()``, will be considered relative to the value\n\
defined by the ``cwd`` property of this object.\n\
");

static PyObject* PyBobIoHDF5File_HasGroup(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  try {
    if (self->f->hasGroup(path)) Py_RETURN_TRUE;
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could not check for group `%s' in file `%s': %s", path, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while checking for group `%s' in HDF5 file `%s'", path, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_FALSE;
}

PyDoc_STRVAR(s_has_group_str, "has_group");
PyDoc_STRVAR(s_has_group_doc,
"x.has_group(path) -> bool\n\
\n\
Checks if a path (group) exists inside a file\n\
\n\
Parameters:\n\
\n\
path\n\
  [str] The path to check\n\
\n\
Checks if a path (i.e. a *group* in HDF5 parlance) exists inside\n\
a file. This method does not work for datasets, only for\n\
directories. If the given path is relative, it is take w.r.t.\n\
to the current working directory.\n\
");

static PyObject* PyBobIoHDF5File_CreateGroup(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* path = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &path)) return 0;

  try {
    self->f->createGroup(path);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could not create group `%s' in file `%s': %s", path, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while creating group `%s' in HDF5 file `%s'", path, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_create_group_str, "create_group");
PyDoc_STRVAR(s_create_group_doc,
"x.create_group(path) -> None\n\
\n\
Creates a new path (group) inside the file.\n\
\n\
Parameters:\n\
\n\
path\n\
  [str] The path to check\n\
\n\
Creates a new directory (i.e., a *group* in HDF5 parlance) inside\n\
the file. A relative path is taken w.r.t. to the current\n\
directory. If the directory already exists (check it with\n\
:py:meth:`HDF5File.has_group()`, an exception will be raised.\n\
");

static PyObject* PyBobIoHDF5File_HasDataset(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  try {
    if (self->f->contains(key)) Py_RETURN_TRUE;
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could not check for dataset `%s' in file `%s': %s", key, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while checking for dataset `%s' in HDF5 file `%s'", key, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_FALSE;
}

PyDoc_STRVAR(s_has_key_str, "has_key");
PyDoc_STRVAR(s_has_dataset_str, "has_dataset");
PyDoc_STRVAR(s_has_dataset_doc,
"x.has_dataset(key) -> bool\n\
\n\
Checks if a dataset exists inside a file\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The dataset path to check\n\
\n\
Checks if a dataset exists inside a file, on the specified path.\n\
If the given path is relative, it is take w.r.t. to the current\n\
working directory.\n\
");

static int PyBobIo_H5AsTypenum (bob::io::hdf5type type) {

  switch(type) {
    case bob::io::s:
      return NPY_STRING;
    case bob::io::b:
      return NPY_BOOL;
    case bob::io::i8:
      return NPY_INT8;
    case bob::io::i16:
      return NPY_INT16;
    case bob::io::i32:
      return NPY_INT32;
    case bob::io::i64:
      return NPY_INT64;
    case bob::io::u8:
      return NPY_UINT8;
    case bob::io::u16:
      return NPY_UINT16;
    case bob::io::u32:
      return NPY_UINT32;
    case bob::io::u64:
      return NPY_UINT64;
    case bob::io::f32:
      return NPY_FLOAT32;
    case bob::io::f64:
      return NPY_FLOAT64;
#ifdef NPY_FLOAT128
    case bob::io::f128:
      return NPY_FLOAT128;
#endif
    case bob::io::c64:
      return NPY_COMPLEX64;
    case bob::io::c128:
      return NPY_COMPLEX128;
#ifdef NPY_COMPLEX256
    case bob::io::c256:
      return NPY_COMPLEX256;
#endif
    default:
      PyErr_Format(PyExc_TypeError, "unsupported Bob/C++ element type (%s)", bob::io::stringize(type));
  }

  return NPY_NOTYPE;

}

static PyObject* PyBobIo_HDF5TypeAsTuple (const bob::io::HDF5Type& t) {
  const bob::io::HDF5Shape& sh = t.shape();
  size_t ndim = sh.n();
  const hsize_t* shptr = sh.get();

  int type_num = PyBobIo_H5AsTypenum(t.type());
  if (type_num == NPY_NOTYPE) return 0;
  PyObject* dtype = reinterpret_cast<PyObject*>(PyArray_DescrFromType(type_num));
  if (!dtype) return 0;

  PyObject* retval = Py_BuildValue("NN", dtype, PyTuple_New(ndim));
  if (!retval) {
    Py_DECREF(dtype);
    return 0;
  }

  PyObject* shape = PyTuple_GET_ITEM(retval, 1);
  for (Py_ssize_t i=0; i<ndim; ++i) {
    PyTuple_SET_ITEM(shape, i, Py_BuildValue("n", shptr[i]));
  }

  return retval;
}

static PyObject* PyBobIo_HDF5DescriptorAsTuple (const bob::io::HDF5Descriptor& d) {

  PyObject* type = PyBobIo_HDF5TypeAsTuple(d.type);
  if (!type) return 0;
  PyObject* size = Py_BuildValue("n", d.size);
  if (!type) return 0;
  PyObject* expand = d.expandable? Py_True : Py_False;
  Py_INCREF(expand);

  return Py_BuildValue("NNN", type, size, expand);

}

static PyObject* PyBobIoHDF5File_Describe(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  PyObject* retval = 0;

  try {
    const std::vector<bob::io::HDF5Descriptor>& dv = self->f->describe(key);
    retval = PyTuple_New(dv.size());
    for (size_t k=0; k<dv.size(); ++k) {
      PyObject* entry = PyBobIo_HDF5DescriptorAsTuple(dv[k]);
      if (!entry) {
        Py_DECREF(retval);
        return 0;
      }
      PyTuple_SET_ITEM(retval, k, entry);
    }
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could not get description for dataset `%s' in file `%s': %s", key, self->f->filename().c_str(), e.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while getting description for dataset `%s' in HDF5 file `%s'", key, self->f->filename().c_str());
  }

  return retval;
}

PyDoc_STRVAR(s_describe_str, "describe");
PyDoc_STRVAR(s_describe_doc,
"x.describe(path) -> tuple\n\
\n\
Describes a dataset type/shape, if it exists inside a file\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The dataset path to describe\n\
\n\
If a given path to an HDF5 dataset exists inside the file,\n\
return a type description of objects recorded in such a dataset,\n\
otherwise, raises an exception. The returned value type is a\n\
tuple of tuples (HDF5Type, number-of-objects, expandable)\n\
describing the capabilities if the file is read using theses\n\
formats.\n\
");

static PyObject* PyBobIoHDF5File_Unlink(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  try {
    self->f->unlink(key);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could unlink dataset `%s' in file `%s': %s", key, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while unlinking dataset `%s' in HDF5 file `%s'", key, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_unlink_str, "unlink");
PyDoc_STRVAR(s_unlink_doc,
"x.unlink(key) -> None\n\
\n\
Unlinks datasets inside the file making them invisible.\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The dataset path to describe\n\
\n\
If a given path to an HDF5 dataset exists inside the file,\n\
unlinks it. Please note this will note remove the data from\n\
the file, just make it inaccessible. If you wish to cleanup,\n\
save the reacheable objects from this file to another HDF5File\n\
object using copy(), for example.\n\
");

static PyObject* PyBobIoHDF5File_Rename(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"from", "to", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* from = 0;
  char* to = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss", kwlist, &from, &to)) return 0;

  try {
    self->f->rename(from, to);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could rename dataset `%s' to `%s' in file `%s': %s", from, to, self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while renaming dataset `%s' to `%s' in HDF5 file `%s'", from, to, self->f->filename().c_str());
    return 0;
  }

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_rename_str, "rename");
PyDoc_STRVAR(s_rename_doc,
"x.rename(from, to) -> None\n\
\n\
Renames datasets in a file\n\
\n\
Parameters:\n\
\n\
from\n\
  [str] The path to the data being renamed\n\
\n\
to\n\
  [str] The new name of the dataset\n\
\n\
");

static PyObject* PyBobIoHDF5File_Paths(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"relative", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* pyrel = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &pyrel)) return 0;

  bool relative = false;
  if (pyrel && PyObject_IsTrue(pyrel)) relative = true;

  PyObject* retval = 0;

  try {
    std::vector<std::string> values;
    self->f->paths(values, relative);
    retval = PyTuple_New(values.size());
    for (size_t i=0; i<values.size(); ++i) {
      PyTuple_SET_ITEM(retval, i, Py_BuildValue("s", values[i].c_str()));
    }
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could read dataset names from file `%s': %s", self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while reading dataset names from HDF5 file `%s'", self->f->filename().c_str());
    return 0;
  }

  return retval;
}

PyDoc_STRVAR(s_keys_str, "keys");
PyDoc_STRVAR(s_paths_str, "paths");
PyDoc_STRVAR(s_paths_doc,
"x.paths([relative=False]) -> tuple\n\
\n\
Lists datasets available inside this file\n\
\n\
Parameters:\n\
\n\
relative\n\
  [bool, optional] if set to ``True``, the returned paths are\n\
  relative to the current working directory, otherwise they are\n\
  absolute.\n\
\n\
Returns all paths to datasets available inside this file, stored\n\
under the current working directory. If relative is set to ``True``,\n\
the returned paths are relative to the current working directory,\n\
otherwise they are absolute.\n\
");

static PyObject* PyBobIoHDF5File_SubGroups(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"relative", "recursive", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* pyrel = 0;
  PyObject* pyrec = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &pyrel, &pyrec)) return 0;

  bool relative = false;
  if (pyrel && PyObject_IsTrue(pyrel)) relative = true;
  bool recursive = true;
  if (pyrec && !PyObject_IsTrue(pyrec)) recursive = false;

  PyObject* retval = 0;

  try {
    std::vector<std::string> values;
    self->f->sub_groups(values, relative, recursive);
    retval = PyTuple_New(values.size());
    for (size_t i=0; i<values.size(); ++i) {
      PyTuple_SET_ITEM(retval, i, Py_BuildValue("s", values[i].c_str()));
    }
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "could read group names from file `%s': %s", self->f->filename().c_str(), e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "unknown exception caught while reading group names from HDF5 file `%s'", self->f->filename().c_str());
    return 0;
  }

  return retval;
}

PyDoc_STRVAR(s_sub_groups_str, "sub_groups");
PyDoc_STRVAR(s_sub_groups_doc,
"x.sub_groups([relative=False, [recursive=True]]) -> tuple\n\
\n\
Lists groups (directories) in the current file.\n\
\n\
Parameters:\n\
\n\
relative\n\
  [bool, optional] if set to ``True``, the returned sub-groups are\n\
  relative to the current working directory, otherwise they are\n\
  absolute.\n\
\n\
recursive\n\
  [bool, optional] if set to ``False``, the returned sub-groups\n\
  are only the ones in the current directory. Otherwise, recurse\n\
  down the directory structure.\n\
\n\
");

static PyObject* PyBobIoHDF5File_Xread(PyBobIoHDF5FileObject* self, 
    const char* p, int descriptor, int pos) {

  const std::vector<bob::io::HDF5Descriptor>& D = self->f->describe(p);

  //last descriptor always contains the full readout.
  const bob::io::HDF5Type& type = D[descriptor].type;
  const bob::io::HDF5Shape& shape = type.shape();

  if (shape.n() == 1 && shape[0] == 1) { //read as scalar
    try {
      switch(type.type()) {
        case bob::io::s:
          return Py_BuildValue("s", self->f->read<std::string>(p, pos).c_str());
        case bob::io::b:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<bool>(p, pos));
        case bob::io::i8:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<int8_t>(p, pos));
        case bob::io::i16:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<int16_t>(p, pos));
        case bob::io::i32:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<int32_t>(p, pos));
        case bob::io::i64:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<int64_t>(p, pos));
        case bob::io::u8:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<uint8_t>(p, pos));
        case bob::io::u16:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<uint16_t>(p, pos));
        case bob::io::u32:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<uint32_t>(p, pos));
        case bob::io::u64:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<uint64_t>(p, pos));
        case bob::io::f32:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<float>(p, pos));
        case bob::io::f64:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<double>(p, pos));
        case bob::io::f128:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<long double>(p, pos));
        case bob::io::c64:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<float> >(p, pos));
        case bob::io::c128:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<double> >(p, pos));
        case bob::io::c256:
          return PyBlitzArrayCxx_FromCScalar(self->f->read<std::complex<long double> >(p, pos));
        default:
          PyErr_Format(PyExc_TypeError, "unsupported HDF5 type: %s", type.str().c_str());
          return 0;
      }
    }
    catch (std::exception& e) {
      PyErr_Format(PyExc_RuntimeError, "cannot read %s scalar from dataset `%s' at position %d from HDF5 file `%s': %s", bob::io::stringize(type.type()), p, pos, self->f->filename().c_str(), e.what());
      return 0;
    }
    catch (...) {
      PyErr_Format(PyExc_RuntimeError, "caught unknown exception while reading %s scalar from dataset `%s' at position %d from HDF5 file `%s'", bob::io::stringize(type.type()), p, pos, self->f->filename().c_str());
      return 0;
    }
  }

  //read as an numpy array
  bob::core::array::typeinfo atype;
  type.copy_to(atype);

  int type_num = PyBobIo_AsTypenum(atype.dtype);
  if (type_num == NPY_NOTYPE) return 0; ///< failure

  npy_intp pyshape[NPY_MAXDIMS];
  for (int k=0; k<atype.nd; ++k) pyshape[k] = atype.shape[k];

  PyObject* retval = PyArray_SimpleNew(atype.nd, pyshape, type_num);
  if (!retval) return 0;

  try {
    self->f->read_buffer(p, pos, atype, PyArray_DATA((PyArrayObject*)retval));
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "cannot read dataset `%s' at position %d with descriptor `%s' from HDF5 file `%s': %s", p, pos, type.str().c_str(), self->f->filename().c_str(), e.what());
    Py_DECREF(retval);
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "caught unknown exception while reading dataset `%s' at position %d with descriptor `%s' from HDF5 file `%s'", p, pos, type.str().c_str(), self->f->filename().c_str());
    Py_DECREF(retval);
    return 0;
  }

  return retval;
}

static PyObject* PyBobIoHDF5File_Read(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  const char* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &key)) return 0;

  return PyBobIoHDF5File_Xread(self, key, 1, 0);

}

PyDoc_STRVAR(s_read_str, "read");
PyDoc_STRVAR(s_read_doc,
"x.read(key, [pos=-1]) -> numpy.ndarray\n\
\n\
Reads whole datasets from the file.\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The path to the dataset to read data from. Can be\n\
  an absolute value (starting with a leading ``'/'``) or\n\
  relative to the current working directory (``cwd``).\n\
\n\
");

static PyObject* PyBobIoHDF5File_ListRead(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", "pos", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  const char* key = 0;
  Py_ssize_t pos = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|n", kwlist, &key, &pos)) return 0;

  if (pos >= 0) return PyBobIoHDF5File_Xread(self, key, 1, pos);

  //otherwise returns as a list
  const std::vector<bob::io::HDF5Descriptor>& D = self->f->describe(key);

  PyObject* retval = PyTuple_New(D[0].size);
  if (!retval) return 0;

  for (uint64_t k=0; k<D[0].size; ++k) {
    PyObject* item = PyBobIoHDF5File_Xread(self, key, 0, k);
    if (!item) {
      Py_DECREF(retval);
      return 0;
    }
    PyTuple_SET_ITEM(retval, k, item);
  }

  return retval;

}

PyDoc_STRVAR(s_lread_str, "lread");
PyDoc_STRVAR(s_lread_doc,
"x.lread(key, [pos=-1]) -> list|numpy.ndarray\n\
\n\
Reads some contents of the dataset.\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The path to the dataset to read data from. Can be\n\
  an absolute value (starting with a leading ``'/'``) or\n\
  relative to the current working directory (``cwd``).\n\
\n\
pos\n\
  [int, optional] Returns a single object if ``pos`` >= 0,\n\
  otherwise a list by reading all objects in sequence.\n\
\n\
This method reads contents from a dataset, treating the\n\
N-dimensional dataset like a container for multiple objects\n\
with N-1 dimensions. It returns a single\n\
:py:class:`numpy.ndarray` in case ``pos`` is set to a\n\
value >= 0, or a list of arrays otherwise.\n\
");

static PyObject* PyBobIoHDF5File_Replace(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", "pos", "data", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* key = 0;
  Py_ssize_t pos = -1;
  PyObject* data = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OnO", kwlist, &key, &pos, &data)) return 0;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_replace_str, "replace");
PyDoc_STRVAR(s_replace_doc,
"x.replace(path, pos, data) -> None\n\
\n\
Modifies the value of a scalar/array in a dataset.\n\
\n\
Parameters:\n\
\n\
key\n\
  [str] The path to the dataset to read data from. Can be\n\
  an absolute value (starting with a leading ``'/'``) or\n\
  relative to the current working directory (``cwd``).\n\
\n\
pos\n\
  [int] Position, within the dataset, of the object to be\n\
  replaced. The object position on the dataset must exist,\n\
  or an exception is raised.\n\
\n\
data\n\
  [scalar|numpy.ndarray] Object to replace the value with.\n\
  This value must be compatible with the typing information\n\
  on the dataset, or an exception will be raised.\n\
\n\
");

static PyObject* PyBobIoHDF5File_Append(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", "data", "compression", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* key = 0;
  PyObject* data = 0;
  Py_ssize_t compression = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist, &key, &data, &compression)) return 0;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_append_str, "append");
PyDoc_STRVAR(s_append_doc,
"x.append(path, data, [compression=0]) -> None\n\
\n\
Appends a scalar or an array to a dataset\n\
\n\
Parameters:\n\
\n\
path\n\
  [str] The path to the dataset to read data from. Can be\n\
  an absolute value (starting with a leading ``'/'``) or\n\
  relative to the current working directory (``cwd``).\n\
\n\
data\n\
  [scalar|numpy.ndarray] Object to append to the dataset.\n\
  This value must be compatible with the typing information\n\
  on the dataset, or an exception will be raised.\n\
  You can also, optionally, set this to an iterable of\n\
  scalars or arrays. This will cause this method to iterate\n\
  over the elements and add each individually.\n\
\n\
compression\n\
  This parameter is effective when appending arrays. Set this\n\
  to a number betwen 0 (default) and 9 (maximum) to compress\n\
  the contents of this dataset. This setting is only effective\n\
  if the dataset does not yet exist, otherwise, the previous\n\
  setting is respected.\n\
\n\
");

static PyObject* PyBobIoHDF5File_Set(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"path", "data", "compression", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* key = 0;
  PyObject* data = 0;
  Py_ssize_t compression = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist, &key, &data, &compression)) return 0;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_set_str, "set");
PyDoc_STRVAR(s_set_doc,
"x.set(path, data, [compression=0]) -> None\n\
\n\
Sets the scalar or array at position 0 to the given value.\n\
\n\
Parameters:\n\
\n\
path\n\
  [str] The path to the dataset to read data from. Can be\n\
  an absolute value (starting with a leading ``'/'``) or\n\
  relative to the current working directory (``cwd``).\n\
\n\
data\n\
  [scalar|numpy.ndarray] Object to append to the dataset.\n\
  This value must be compatible with the typing information\n\
  on the dataset, or an exception will be raised.\n\
  You can also, optionally, set this to an iterable of\n\
  scalars or arrays. This will cause this method to iterate\n\
  over the elements and add each individually.\n\
\n\
compression\n\
  This parameter is effective when appending arrays. Set this\n\
  to a number betwen 0 (default) and 9 (maximum) to compress\n\
  the contents of this dataset. This setting is only effective\n\
  if the dataset does not yet exist, otherwise, the previous\n\
  setting is respected.\n\
\n\
This method is equivalent to checking if the scalar or array at\n\
position 0 exists and then replacing it. If the path does not\n\
exist, we append the new scalar or array.\n\
");

static PyObject* PyBobIoHDF5File_Copy(PyBobIoHDF5FileObject* self, PyObject *args, PyObject* kwds) {
  
  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"file", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBobIoHDF5FileObject* file = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, &PyBobIoHDF5File_Converter, &file)) return 0;

  Py_RETURN_NONE;
}

PyDoc_STRVAR(s_copy_str, "copy");
PyDoc_STRVAR(s_copy_doc,
"x.copy(file) -> None\n\
\n\
Copies all accessible content to another HDF5 file\n\
\n\
Parameters:\n\
\n\
file\n\
  [HDF5File] The file (already opened), to copy the contents to.\n\
  Unlinked contents of this file will not be copied. This can be\n\
  used as a method to trim unwanted content in a file.\n\
\n\
");

static PyMethodDef PyBobIoHDF5File_Methods[] = {
  {
    s_cd_str,
    (PyCFunction)PyBobIoHDF5File_ChangeDirectory,
    METH_VARARGS|METH_KEYWORDS,
    s_cd_doc,
  },
  {
    s_has_group_str,
    (PyCFunction)PyBobIoHDF5File_HasGroup,
    METH_VARARGS|METH_KEYWORDS,
    s_has_group_doc,
  },
  {
    s_create_group_str,
    (PyCFunction)PyBobIoHDF5File_CreateGroup,
    METH_VARARGS|METH_KEYWORDS,
    s_create_group_doc,
  },
  {
    s_has_dataset_str,
    (PyCFunction)PyBobIoHDF5File_HasDataset,
    METH_VARARGS|METH_KEYWORDS,
    s_has_dataset_doc,
  },
  {
    s_has_key_str,
    (PyCFunction)PyBobIoHDF5File_HasDataset,
    METH_VARARGS|METH_KEYWORDS,
    s_has_dataset_doc,
  },
  {
    s_describe_str,
    (PyCFunction)PyBobIoHDF5File_Describe,
    METH_VARARGS|METH_KEYWORDS,
    s_describe_doc,
  },
  {
    s_unlink_str,
    (PyCFunction)PyBobIoHDF5File_Unlink,
    METH_VARARGS|METH_KEYWORDS,
    s_unlink_doc,
  },
  {
    s_rename_str,
    (PyCFunction)PyBobIoHDF5File_Rename,
    METH_VARARGS|METH_KEYWORDS,
    s_rename_doc,
  },
  {
    s_paths_str,
    (PyCFunction)PyBobIoHDF5File_Paths,
    METH_VARARGS|METH_KEYWORDS,
    s_paths_doc,
  },
  {
    s_keys_str,
    (PyCFunction)PyBobIoHDF5File_Paths,
    METH_VARARGS|METH_KEYWORDS,
    s_paths_doc,
  },
  {
    s_sub_groups_str,
    (PyCFunction)PyBobIoHDF5File_SubGroups,
    METH_VARARGS|METH_KEYWORDS,
    s_sub_groups_doc,
  },
  {
    s_read_str,
    (PyCFunction)PyBobIoHDF5File_Read,
    METH_VARARGS|METH_KEYWORDS,
    s_read_doc,
  },
  {
    s_lread_str,
    (PyCFunction)PyBobIoHDF5File_ListRead,
    METH_VARARGS|METH_KEYWORDS,
    s_lread_doc,
  },
  {
    s_replace_str,
    (PyCFunction)PyBobIoHDF5File_Replace,
    METH_VARARGS|METH_KEYWORDS,
    s_replace_doc,
  },
  {
    s_append_str,
    (PyCFunction)PyBobIoHDF5File_Append,
    METH_VARARGS|METH_KEYWORDS,
    s_append_doc,
  },
  {
    s_set_str,
    (PyCFunction)PyBobIoHDF5File_Set,
    METH_VARARGS|METH_KEYWORDS,
    s_set_doc,
  },
  {
    s_copy_str,
    (PyCFunction)PyBobIoHDF5File_Copy,
    METH_VARARGS|METH_KEYWORDS,
    s_copy_doc,
  },
  {0}  /* Sentinel */
};

/**

    .def("get_attributes", &hdf5file_get_attributes, hdf5file_get_attributes_overloads((arg("self"), arg("path")="."), "Returns a dictionary containing all attributes related to a particular (existing) path in this file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))

    .def("get_attribute", &hdf5file_get_attribute, hdf5file_get_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Returns an object representing an attribute attached to a particular (existing) path in this file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))

    .def("set_attributes", &hdf5file_set_attributes, hdf5file_set_attributes_overloads((arg("self"), arg("attrs"), arg("path")="."), "Sets attributes in a given (existing) path using a dictionary containing the names (keys) and values of those attributes. The path may point to a subdirectory or to a particular dataset. Only simple scalars (booleans, integers, floats and complex numbers) and arrays of those are supported at the time being. You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`). If the path does not exist, a RuntimeError is raised."))

    .def("set_attribute", &hdf5file_set_attribute, hdf5file_set_attribute_overloads((arg("self"), arg("name"), arg("value"), arg("path")="."), "Sets the attribute in a given (existing) path using the value provided. The path may point to a subdirectory or to a particular dataset. Only simple scalars (booleans, integers, floats and complex numbers) and arrays of those are supported at the time being. You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`). If the path does not exist, a RuntimeError is raised."))

    .def("has_attribute", &hdf5file_has_attribute, hdf5file_has_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Checks if given attribute exists in a given (existing) path. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))

    .def("delete_attribute", &hdf5file_del_attribute, hdf5file_del_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Deletes a given attribute associated to a (existing) path in the file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))

    .def("delete_attributes", &hdf5file_del_attributes, hdf5file_del_attributes_overloads((arg("self"), arg("path")="."), "Deletes **all** attributes associated to a (existing) path in the file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))

**/

PyTypeObject PyBobIoHDF5File_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_hdf5file_str,                             /*tp_name*/
    sizeof(PyBobIoHDF5FileObject),              /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobIoHDF5File_Delete,         /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBobIoHDF5File_Repr,             /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0, //&PyBobIoHDF5File_Mapping,                   /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBobIoHDF5File_Repr,             /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_hdf5file_doc,                             /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBobIoHDF5File_Methods,                    /* tp_methods */
    0,                                          /* tp_members */
    0, //PyBobIoHDF5File_getseters,                  /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobIoHDF5File_Init,             /* tp_init */
    0,                                          /* tp_alloc */
    PyBobIoHDF5File_New,                        /* tp_new */
};

/**

    .add_property("cwd", &bob::io::HDF5File::cwd)

    .def("__contains__", &bob::io::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")

  //this class describes an HDF5 type
  class_<bob::io::HDF5Type, boost::shared_ptr<bob::io::HDF5Type> >("HDF5Type", "Support to compare data types, convert types into runtime equivalents and make our life easier when deciding what to input and output.", no_init)
    .def("__eq__", &bob::io::HDF5Type::operator==)
    .def("__ne__", &bob::io::HDF5Type::operator!=)
#   define DECLARE_SUPPORT(T) .def("compatible", &bob::io::HDF5Type::compatible<T>, (arg("self"), arg("value")), "Tests compatibility of this type against a given scalar")
    DECLARE_SUPPORT(bool)
    DECLARE_SUPPORT(int8_t)
    DECLARE_SUPPORT(int16_t)
    DECLARE_SUPPORT(int32_t)
    DECLARE_SUPPORT(int64_t)
    DECLARE_SUPPORT(uint8_t)
    DECLARE_SUPPORT(uint16_t)
    DECLARE_SUPPORT(uint32_t)
    DECLARE_SUPPORT(uint64_t)
    DECLARE_SUPPORT(float)
    DECLARE_SUPPORT(double)
    //DECLARE_SUPPORT(long double)
    DECLARE_SUPPORT(std::complex<float>)
    DECLARE_SUPPORT(std::complex<double>)
    //DECLARE_SUPPORT(std::complex<long double>)
    DECLARE_SUPPORT(std::string)
#   undef DECLARE_SUPPORT
    .def("compatible", &hdf5type_compatible, (arg("self"), arg("array")), "Tests compatibility of this type against a given array")
    .def("shape", &hdf5type_shape, (arg("self")), "Returns the shape of the elements described by this type")
    .def("type_str", &bob::io::HDF5Type::type_str, (arg("self")), "Returns a stringified representation of the base element type")
    .def("element_type", &bob::io::HDF5Type::element_type, (arg("self")), "Returns a representation of the element type one of the bob supported element types.")
    ;

  //defines the descriptions returned by HDF5File::describe()
  class_<bob::io::HDF5Descriptor, boost::shared_ptr<bob::io::HDF5Descriptor> >("HDF5Descriptor", "A dataset descriptor describes one of the possible ways to read a dataset", no_init)
    .def_readonly("type", &bob::io::HDF5Descriptor::type)
    .def_readonly("size", &bob::io::HDF5Descriptor::size)
    .def_readonly("expandable", &bob::io::HDF5Descriptor::expandable)
    ;
**/
