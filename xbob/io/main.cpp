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
#include <xbob.blitz/capi.h>
#include <xbob.blitz/cleanup.h>

static PyMethodDef module_methods[] = {
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::io classes and methods");

int PyXbobIo_APIVersion = XBOB_IO_API_VERSION;

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

  PyBobIoFile_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoFile_Type) < 0) return 0;

  PyBobIoFileIterator_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoFileIterator_Type) < 0) return 0;

  PyBobIoHDF5File_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoHDF5File_Type) < 0) return 0;

#if WITH_FFMPEG
  PyBobIoVideoReader_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoVideoReader_Type) < 0) return 0;

  PyBobIoVideoReaderIterator_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoVideoReaderIterator_Type) < 0) return 0;

  PyBobIoVideoWriter_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobIoVideoWriter_Type) < 0) return 0;
#endif /* WITH_FFMPEG */

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m);

  /* register some constants */
  if (PyModule_AddIntConstant(m, "__api_version__", XBOB_IO_API_VERSION) < 0) return 0;
  if (PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION) < 0) return 0;

  /* register the types to python */
  Py_INCREF(&PyBobIoFile_Type);
  if (PyModule_AddObject(m, "File", (PyObject *)&PyBobIoFile_Type) < 0) return 0;

  Py_INCREF(&PyBobIoFileIterator_Type);
  if (PyModule_AddObject(m, "File.iter", (PyObject *)&PyBobIoFileIterator_Type) < 0) return 0;

  Py_INCREF(&PyBobIoHDF5File_Type);
  if (PyModule_AddObject(m, "HDF5File", (PyObject *)&PyBobIoHDF5File_Type) < 0) return 0;

#if WITH_FFMPEG
  Py_INCREF(&PyBobIoVideoReader_Type);
  if (PyModule_AddObject(m, "VideoReader", (PyObject *)&PyBobIoVideoReader_Type) < 0) return 0;

  Py_INCREF(&PyBobIoVideoReaderIterator_Type);
  if (PyModule_AddObject(m, "VideoReader.iter", (PyObject *)&PyBobIoVideoReaderIterator_Type) < 0) return 0;

  Py_INCREF(&PyBobIoVideoWriter_Type);
  if (PyModule_AddObject(m, "VideoWriter", (PyObject *)&PyBobIoVideoWriter_Type) < 0) return 0;
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

  PyXbobIo_API[PyBobIoFileIterator_Type_NUM] = (void *)&PyBobIoFileIterator_Type;

  /************************
   * I/O generic bindings *
   ************************/
  
  PyXbobIo_API[PyBobIo_AsTypenum_NUM] = (void *)PyBobIo_AsTypenum;

  PyXbobIo_API[PyBobIo_TypeInfoAsTuple_NUM] = (void *)PyBobIo_TypeInfoAsTuple;

  PyXbobIo_API[PyBobIo_FilenameConverter_NUM] = (void *)PyBobIo_FilenameConverter;

  /*****************
   * HDF5 bindings *
   *****************/

  PyXbobIo_API[PyBobIoHDF5File_Type_NUM] = (void *)&PyBobIoHDF5File_Type;
  
  PyXbobIo_API[PyBobIoHDF5File_Check_NUM] = (void *)&PyBobIoHDF5File_Check;

  PyXbobIo_API[PyBobIoHDF5File_Converter_NUM] = (void *)&PyBobIoHDF5File_Converter;

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

  PyXbobIo_API[PyBobIoVideoReader_Type_NUM] = (void *)&PyBobIoVideoReader_Type;

  PyXbobIo_API[PyBobIoVideoReaderIterator_Type_NUM] = (void *)&PyBobIoVideoReaderIterator_Type;

  PyXbobIo_API[PyBobIoVideoWriter_Type_NUM] = (void *)&PyBobIoVideoWriter_Type;
#endif /* WITH_FFMPEG */

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyXbobIo_API,
      XBOB_EXT_MODULE_PREFIX "." XBOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyXbobIo_API, 0);

#endif

  if (!c_api_object) return 0;

  if (PyModule_AddObject(m, "_C_API", c_api_object) < 0) return 0;

  /* imports xbob.blitz C-API + dependencies */
  if (import_xbob_blitz() < 0) return 0;

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
