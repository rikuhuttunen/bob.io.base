/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::io
 */

#define XBOB_IO_MODULE
#include <xbob.io/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <blitz.array/capi.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::io classes and methods");

int PyXbobIo_APIVersion = XBOB_IO_API_VERSION;

#define ENTRY_FUNCTION_INNER(a) init ## a
#define ENTRY_FUNCTION(a) ENTRY_FUNCTION_INNER(a)

PyMODINIT_FUNC ENTRY_FUNCTION(XBOB_IO_MODULE_NAME) (void) {

  PyBobIoFile_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoFile_Type) < 0) return;

#if WITH_FFMPEG
  PyBobIoVideoReader_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoVideoReader_Type) < 0) return;
#endif /* WITH_FFMPEG */

  PyObject* m = Py_InitModule3(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME),
      module_methods, module_docstr);

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_IO_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", BOOST_PP_STRINGIZE(XBOB_IO_VERSION));

  /* register the types to python */
  Py_INCREF(&PyBobIoFile_Type);
  PyModule_AddObject(m, "File", (PyObject *)&PyBobIoFile_Type);

#if WITH_FFMPEG
  Py_INCREF(&PyBobIoVideoReader_Type);
  PyModule_AddObject(m, "VideoReader", (PyObject *)&PyBobIoVideoReader_Type);
#endif /* WITH_FFMPEG */

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

  PyXbobIo_API[PyBobIo_TypeInfoAsTuple_NUM] = (void *)PyBobIo_TypeInfoAsTuple;

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

  PyXbobIo_API[PyBobIoVideoReader_Type_NUM] = (void *)&PyBobIoVideoReader_Type;
#endif /* WITH_FFMPEG */

  /* imports the NumPy C-API */
  import_array();

  /* imports blitz.array C-API */
  import_blitz_array();

}
