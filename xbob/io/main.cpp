/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::io
 */

#define XBOB_IO_MODULE
#include <xbob.io/api.h>
#include <bob/io/CodecRegistry.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <blitz.array/capi.h>

static PyObject* PyBobIo_Extensions(PyObject*) {

  typedef std::map<std::string, std::string> map_type;
  const map_type& table = bob::io::CodecRegistry::getExtensions();

  PyObject* retval = PyDict_New();
  if (!retval) return 0;

  for (auto it=table.begin(); it!=table.end(); ++it) {
#   if PY_VERSION_HEX >= 0x03000000
    PyObject* value = PyString_FromString(it->second.c_str());
#   else
    PyObject* value = PyUnicode_FromString(it->second.c_str());
#   endif
    if (!value) {
      Py_DECREF(retval);
      return 0;
    }
    PyDict_SetItemString(retval, it->first.c_str(), value);
    Py_DECREF(value);
  }
  return retval;

}

PyDoc_STRVAR(s_extensions_str, "extensions");
PyDoc_STRVAR(s_extensions_doc,
"as_blitz(x) -> dict\n\
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

PyDoc_STRVAR(module_docstr, "bob::io classes and methods");

int PyXbobIo_APIVersion = XBOB_IO_API_VERSION;

#define ENTRY_FUNCTION_INNER(a) init ## a
#define ENTRY_FUNCTION(a) ENTRY_FUNCTION_INNER(a)

PyMODINIT_FUNC ENTRY_FUNCTION(XBOB_IO_MODULE_NAME) (void) {

  PyBobIoFile_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoFile_Type) < 0) return;

  PyObject* m = Py_InitModule3(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME),
      module_methods, module_docstr);

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_IO_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", BOOST_PP_STRINGIZE(XBOB_IO_VERSION));

  /* register the types to python */
  Py_INCREF(&PyBobIoFile_Type);
  PyModule_AddObject(m, "File", (PyObject *)&PyBobIoFile_Type);

  static void* PyXbobIo_API[PyXbobIo_API_pointers];

  /* exhaustive list of C APIs */

  /**************
   * Versioning *
   **************/

  PyXbobIo_API[PyXbobIo_APIVersion_NUM] = (void *)&PyXbobIo_APIVersion;

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

  PyXbobIo_API[PyBobIoFile_Type_NUM] = (void *)&PyBobIoFile_Type;

  /************************
   * I/O generic bindings *
   ************************/
  
  PyXbobIo_API[PyBobIo_AsTypenum_NUM] = (void *)PyBobIo_AsTypenum;

  /* imports the NumPy C-API */
  import_array();

  /* imports blitz.array C-API */
  import_blitz_array();

}
