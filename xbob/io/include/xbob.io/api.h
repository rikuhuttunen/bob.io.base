/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013 
 *
 * @brief C/C++ API for bob::io
 */

#ifndef XBOB_IO_H
#define XBOB_IO_H

#include <xbob.io/config.h>
#include <bob/config.h>
#include <bob/io/File.h>

#if WITH_FFMPEG
#include <bob/io/VideoReader.h>
#endif /* WITH_FFMPEG */

#include <boost/preprocessor/stringize.hpp>
#include <boost/shared_ptr.hpp>
#include <Python.h>

#define XBOB_IO_MODULE_PREFIX xbob.io
#define XBOB_IO_MODULE_NAME _library

/*******************
 * C API functions *
 *******************/

/**************
 * Versioning *
 **************/

#define PyXbobIo_APIVersion_NUM 0
#define PyXbobIo_APIVersion_TYPE int

/*****************************
 * Bindings for xbob.io.file *
 *****************************/

/* Type definition for PyBobIoFileObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::File> f;

} PyBobIoFileObject;

#define PyBobIoFile_Type_NUM 1
#define PyBobIoFile_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD
  
  /* Type-specific fields go here. */
  PyBobIoFileObject* pyfile;
  Py_ssize_t curpos;

} PyBobIoFileIteratorObject;

#define PyBobIoFileIterator_Type_NUM 2
#define PyBobIoFileIterator_Type_TYPE PyTypeObject

/************************
 * I/O generic bindings *
 ************************/

#define PyBobIo_AsTypenum_NUM 3
#define PyBobIo_AsTypenum_RET int
#define PyBobIo_AsTypenum_PROTO (bob::core::array::ElementType et)

#define PyBobIo_TypeInfoAsTuple_NUM 4
#define PyBobIo_TypeInfoAsTuple_RET PyObject*
#define PyBobIo_TypeInfoAsTuple_PROTO (const bob::core::array::typeinfo& ti)

#if WITH_FFMPEG

/******************
 * Video bindings *
 ******************/

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::VideoReader> v;

} PyBobIoVideoReaderObject;

#define PyBobIoVideoReader_Type_NUM 5
#define PyBobIoVideoReader_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  PyBobIoVideoReaderObject* pyreader;
  boost::shared_ptr<bob::io::VideoReader::const_iterator> iter;

} PyBobIoVideoReaderIteratorObject;

#define PyBobIoVideoReaderIterator_Type_NUM 5
#define PyBobIoVideoReaderIterator_Type_TYPE PyTypeObject

#endif /* WITH_FFMPEG */

/* Total number of C API pointers */
#if WITH_FFMPEG
#  define PyXbobIo_API_pointers 6
#else
#  define PyXbobIo_API_pointers 7
#endif /* WITH_FFMPEG */

#ifdef XBOB_IO_MODULE

  /* This section is used when compiling `xbob.core.random' itself */

  /**************
   * Versioning *
   **************/

  extern int PyXbobIo_APIVersion;

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

  extern PyBobIoFile_Type_TYPE PyBobIoFile_Type;
  extern PyBobIoFileIterator_Type_TYPE PyBobIoFileIterator_Type;

  /************************
   * I/O generic bindings *
   ************************/

  PyBobIo_AsTypenum_RET PyBobIo_AsTypenum PyBobIo_AsTypenum_PROTO;
  
  PyBobIo_TypeInfoAsTuple_RET PyBobIo_TypeInfoAsTuple PyBobIo_TypeInfoAsTuple_PROTO;

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

  extern PyBobIoVideoReader_Type_TYPE PyBobIoVideoReader_Type;
  extern PyBobIoVideoReaderIterator_Type_TYPE PyBobIoVideoReaderIterator_Type;
#endif /* WITH_FFMPEG */

#else

  /* This section is used in modules that use `blitz.array's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define XBOB_IO_MAKE_API_NAME_INNER(a) XBOB_IO_ ## a
#    define XBOB_IO_MAKE_API_NAME(a) XBOB_IO_MAKE_API_NAME_INNER(a)
#    define PyXbobIo_API XBOB_IO_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobIo_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobIo_API;
#    else
  static void **PyXbobIo_API=NULL;
#    endif
#  endif

  static void **PyXbobIo_API;

  /**************
   * Versioning *
   **************/

# define PyXbobIo_APIVersion (*(PyXbobIo_APIVersion_TYPE *)PyXbobIo_API[PyXbobIo_APIVersion_NUM])

  /*****************************
   * Bindings for xbob.io.file *
   *****************************/

# define PyBobIoFile_Type (*(PyBobIoFile_Type_TYPE *)PyXbobIo_API[PyBobIoFile_Type_NUM])
# define PyBobIoFileIterator_Type (*(PyBobIoFileIterator_Type_TYPE *)PyXbobIo_API[PyBobIoFileIterator_Type_NUM])

  /************************
   * I/O generic bindings *
   ************************/

# define PyBobIo_AsTypenum (*(PyBobIo_AsTypenum_RET (*)PyBobIo_AsTypenum_PROTO) PyXbobIo_API[PyBobIo_AsTypenum_NUM])

# define PyBobIo_TypeInfoAsTuple (*(PyBobIo_TypeInfoAsTuple_RET (*)PyBobIo_TypeInfoAsTuple_PROTO) PyXbobIo_API[PyBobIo_TypeInfoAsTuple_NUM])

#if WITH_FFMPEG
  /******************
   * Video bindings *
   ******************/

# define PyBobIoVideoReader_Type (*(PyBobIoVideoReader_Type_TYPE *)PyXbobIo_API[PyBobIoVideoReader_Type_NUM])

# define PyBobIoVideoReaderIterator_Type (*(PyBobIoVideoReaderIterator_Type_TYPE *)PyXbobIo_API[PyBobIoVideoReaderIterator_Type_NUM])
#endif /* WITH_FFMPEG */

  /**
   * Returns -1 on error, 0 on success. PyCapsule_Import will set an exception
   * if there's an error.
   */
  static int import_xbob_io(void) {

#if PY_VERSION_HEX >= 0x02070000

    /* New Python API support for library loading */

    PyXbobIo_API = (void **)PyCapsule_Import(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME) "._C_API", 0);

    if (!PyXbobIo_API) return -1;

#else

    /* Old-style Python API support for library loading */

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME));

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

    if (PyCObject_Check(c_api_object)) {
      PyXbobIo_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }

    Py_DECREF(c_api_object);
    Py_DECREF(module);

#endif
    
    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobIo_API[PyIo_APIVersion_NUM];

    if (XBOB_IO_API_VERSION != imported_version) {
      PyErr_Format(PyExc_RuntimeError, "%s.%s import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX), BOOST_PP_STRINGIZE(XBOB_IO_MODULE_NAME), XBOB_IO_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

#endif /* XBOB_IO_MODULE */

#endif /* XBOB_IO_H */
