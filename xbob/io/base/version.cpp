/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  7 Nov 13:50:16 2013
 *
 * @brief Binds configuration information available from bob
 */

#include <Python.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#define XBOB_IO_BASE_MODULE
#include <xbob.io.base/config.h>

#include <string>
#include <cstdlib>
#include <boost/preprocessor/stringize.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>

#include <bob/config.h>
#include <bob/io/CodecRegistry.h>

#include <xbob.blitz/capi.h>
#include <xbob.blitz/cleanup.h>
#include <hdf5.h>

static int dict_set(PyObject* d, const char* key, const char* value) {
  PyObject* v = Py_BuildValue("s", value);
  if (!v) return 0;
  auto v_ = make_safe(v);
  int retval = PyDict_SetItemString(d, key, v);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

static int dict_steal(PyObject* d, const char* key, PyObject* value) {
  if (!value) return 0;
  auto value_ = make_safe(value);
  int retval = PyDict_SetItemString(d, key, value);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

/**
 * Creates an str object, from a C or C++ string. Returns a **new
 * reference**.
 */
static PyObject* make_object(const char* s) {
  return Py_BuildValue("s", s);
}

/***********************************************************
 * Version number generation
 ***********************************************************/

static PyObject* hdf5_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(H5_VERS_MAJOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_MINOR);
  f % BOOST_PP_STRINGIZE(H5_VERS_RELEASE);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Bob version, API version and platform
 */
static PyObject* bob_version() {
  return Py_BuildValue("sis", BOB_VERSION, BOB_API_VERSION, BOB_PLATFORM);
}

/**
 * Describes the version of Boost libraries installed
 */
static PyObject* boost_version() {
  boost::format f("%d.%d.%d");
  f % (BOOST_VERSION / 100000);
  f % (BOOST_VERSION / 100 % 1000);
  f % (BOOST_VERSION % 100);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Describes the compiler version
 */
static PyObject* compiler_version() {
# if defined(__GNUC__) && !defined(__llvm__)
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(__GNUC__);
  f % BOOST_PP_STRINGIZE(__GNUC_MINOR__);
  f % BOOST_PP_STRINGIZE(__GNUC_PATCHLEVEL__);
  return Py_BuildValue("{ssss}", "name", "gcc", "version", f.str().c_str());
# elif defined(__llvm__) && !defined(__clang__)
  return Py_BuildValue("{ssss}", "name", "llvm-gcc", "version", __VERSION__);
# elif defined(__clang__)
  return Py_BuildValue("{ssss}", "name", "clang", "version", __clang_version__);
# else
  return Py_BuildValue("{ssss}", "name", "unsupported", "version", "unknown");
# endif
}

/**
 * Python version with which we compiled the extensions
 */
static PyObject* python_version() {
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(PY_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(PY_MICRO_VERSION);
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Numpy version
 */
static PyObject* numpy_version() {
  return Py_BuildValue("{ssss}", "abi", BOOST_PP_STRINGIZE(NPY_VERSION),
      "api", BOOST_PP_STRINGIZE(NPY_API_VERSION));
}

/**
 * xbob.blitz c/c++ api version
 */
static PyObject* xbob_blitz_version() {
  return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(XBOB_BLITZ_API_VERSION));
}

static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  if (!dict_steal(retval, "Bob", bob_version())) return 0;
  if (!dict_steal(retval, "HDF5", hdf5_version())) return 0;
  if (!dict_steal(retval, "Boost", boost_version())) return 0;
  if (!dict_steal(retval, "Compiler", compiler_version())) return 0;
  if (!dict_steal(retval, "Python", python_version())) return 0;
  if (!dict_steal(retval, "NumPy", numpy_version())) return 0;
  if (!dict_set(retval, "Blitz++", BZ_VERSION)) return 0;
  if (!dict_steal(retval, "xbob.blitz", xbob_blitz_version())) return 0;

  Py_INCREF(retval);
  Py_INCREF(retval);
  return retval;
}

static PyObject* PyBobIo_Extensions(PyObject*) {

  typedef std::map<std::string, std::string> map_type;
  const map_type& table = bob::io::CodecRegistry::getExtensions();

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto it=table.begin(); it!=table.end(); ++it) {
    PyObject* pyvalue = make_object(it->second.c_str());
    if (!pyvalue) return 0;
    if (PyDict_SetItemString(retval, it->first.c_str(), pyvalue) != 0) {
      return 0;
    }
  }

  Py_INCREF(retval);
  return retval;

}

PyDoc_STRVAR(s_extensions_str, "extensions");
PyDoc_STRVAR(s_extensions_doc,
"extensions() -> dict\n\
\n\
Returns a dictionary containing all extensions and descriptions\n\
currently stored on the global codec registry\n\
");

static PyMethodDef module_methods[] = {
    {
      s_extensions_str,
      (PyCFunction)PyBobIo_Extensions,
      METH_NOARGS,
      s_extensions_doc,
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr,
"Information about software used to compile the C++ Bob API"
);

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  XBOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  /* register version numbers and constants */
  if (PyModule_AddIntConstant(m, "api", XBOB_IO_BASE_API_VERSION) < 0)
    return 0;
  if (PyModule_AddStringConstant(m, "module", XBOB_EXT_MODULE_VERSION) < 0)
    return 0;
  if (PyModule_AddObject(m, "externals", build_version_dictionary()) < 0) return 0;

  /* imports dependencies */
  if (import_xbob_blitz() < 0) {
    PyErr_Print();
    PyErr_Format(PyExc_ImportError, "cannot import `%s'", XBOB_EXT_MODULE_NAME);
    return 0;
  }

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
