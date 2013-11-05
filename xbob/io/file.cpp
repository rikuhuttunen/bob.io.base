/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 11:16:09 2013
 *
 * @brief Bindings to bob::io::File
 */

#define XBOB_IO_MODULE
#include <xbob.io/api.h>
#include <bob/io/CodecRegistry.h>
#include <bob/io/utils.h>
#include <stdexcept>

#define FILETYPE_NAME file
PyDoc_STRVAR(s_file_str, BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(FILETYPE_NAME));

/* How to create a new PyBobIoFileObject */
static PyObject* PyBobIoFile_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoFileObject* self = (PyBobIoFileObject*)type->tp_alloc(type, 0);

  self->f.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoFile_Delete (PyBobIoFileObject* o) {

  o->f.reset();
  o->ob_type->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoFile_Init(PyBobIoFileObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"filename", "mode", "pretend_extension", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* filename = 0;
  char* mode = 0;
  int mode_len = 0;
  char* pretend_extension = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ss|s", kwlist, &filename,
        &mode, &mode_len, &pretend_extension)) return -1;

  if (mode_len != 1 || !(mode[0] != 'r' && mode[0] != 'w' && mode[0] != 'a')) {
    PyErr_Format(PyExc_ValueError, "file open mode string should have 1 element and be either 'r' (read), 'w' (write) or 'a' (append)");
    return -1;
  }

  try {
    if (pretend_extension) {
      self->f = bob::io::open(filename, mode[0], pretend_extension);
    }
    else {
      self->f = bob::io::open(filename, mode[0]);
    }
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "cannot open file `%s' with mode `%s': %s", filename, mode, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot open file `%s' with mode `%s': unknown exception caught", filename, mode);
    return -1;
  }

  return 0; ///< SUCCESS
}

PyDoc_STRVAR(s_file_doc,
"file(filename, mode, [pretend_extension]) -> new bob::io::File\n\
\n\
Use this object to read and write data into files.\n\
\n\
Constructor parameters:\n\
\n\
filename\n\
  [str] The file path to the file you want to open\n\
\n\
mode\n\
  [str] A single character (one of ``'r'``, ``'w'``, ``'a'``),\n\
  indicating if you'd like to read, write or append into the file.\n\
  If you choose ``'w'`` and the file already exists, it will be\n\
  truncated.\n\
\n\
pretend_extension\n\
  [str, optional] Normally we read the file matching the extension\n\
  to one of the available codecs installed with the present release\n\
  of Bob. If you set this parameter though, we will read the file\n\
  as it had a given extension. The value should start with a ``'.'``.\n\
  For example ``'.hdf5'``, to make the file be treated like an HDF5\n\
  file.\n\
\n\
"
);

static PyObject* PyBobIoFile_Repr(PyBobIoFileObject* self) {
  return PyUnicode_FromFormat("%s()", s_file_str);
}

PyTypeObject PyBobIoFile_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_file_str,                                 /*tp_name*/
    sizeof(PyBobIoFileObject),                  /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobIoFile_Delete,             /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    (reprfunc)PyBobIoFile_Repr,                 /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    (reprfunc)PyBobIoFile_Repr,                 /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_file_doc,                                 /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    0, //PyBobIoFile_methods,                               /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobIoFile_Init,                 /* tp_init */
    0,                                          /* tp_alloc */
    PyBobIoFile_New,                            /* tp_new */
};

/**
static object file_read_all(bob::io::File& f) {
  bob::python::py_array a(f.type_all());
  f.read_all(a);
  return a.pyobject(); //shallow copy
}

static object file_read(bob::io::File& f, size_t index) {
  bob::python::py_array a(f.type_all());
  f.read(a, index);
  return a.pyobject(); //shallow copy
}

static boost::shared_ptr<bob::io::File> string_open1 (const std::string& filename,
    const std::string& mode) {
  return bob::io::open(filename, mode[0]);
}

static boost::shared_ptr<bob::io::File> string_open2 (const std::string& filename,
    const std::string& mode, const std::string& pretend_extension) {
  return bob::io::open(filename, mode[0], pretend_extension);
}

static void file_write(bob::io::File& f, object array) {
  bob::python::py_array a(array, object());
  f.write(a);
}

static void file_append(bob::io::File& f, object array) {
  bob::python::py_array a(array, object());
  f.append(a);
}

static dict extensions() {
  typedef std::map<std::string, std::string> map_type;
  dict retval;
  const map_type& table = bob::io::CodecRegistry::getExtensions();
  for (map_type::const_iterator it=table.begin(); it!=table.end(); ++it) {
    retval[it->first] = it->second;
  }
  return retval;
}

void bind_io_file() {

  class_<bob::io::File, boost::shared_ptr<bob::io::File>, boost::noncopyable>("File", "Abstract base class for all Array/Arrayset i/o operations", no_init)
    .def("__init__", make_constructor(string_open1, default_call_policies(), (arg("filename"), arg("mode"))), "Opens a (supported) file for reading arrays. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.")
    .def("__init__", make_constructor(string_open2, default_call_policies(), (arg("filename"), arg("mode"), arg("pretend_extension"))), "Opens a (supported) file for reading arrays but pretends its extension is as given by the last parameter - this way you can, potentially, override the default encoder/decoder used to read and write on the file. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.")
    .add_property("filename", make_function(&bob::io::File::filename, return_value_policy<copy_const_reference>()), "The path to the file being read/written")
    .add_property("type_all", make_function(&bob::io::File::type_all, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("type", make_function(&bob::io::File::type, return_value_policy<copy_const_reference>()), "Typing information to load the file as an Arrayset")
    .add_property("codec_name", make_function(&bob::io::File::name, return_value_policy<copy_const_reference>()), "Name of the File class implementation -- for compatibility reasons with the previous versions of this library")
    .def("read", &file_read_all, (arg("self")), "Reads the whole contents of the file into a NumPy ndarray")
    .def("write", &file_write, (arg("self"), arg("array")), "Writes an array into the file, truncating it first")
    .def("__len__", &bob::io::File::size, (arg("self")), "Size of the file if it is supposed to be read as a set of arrays instead of performing a single read")
    .def("read", &file_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("__getitem__", &file_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("append", &file_append, (arg("self"), arg("array")), "Appends an array to a file. Compatibility requirements may be enforced.")
    ;

  def("extensions", &extensions, "Returns a dictionary containing all extensions and descriptions currently stored on the global codec registry");

}

**/
