/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  5 Nov 12:22:48 2013
 *
 * @brief Python API for bob::io::base
 */

#ifndef BOB_IO_BASE_H
#define BOB_IO_BASE_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use BOB_EXT_* macros here, unfortunately */
#define BOB_IO_BASE_PREFIX    "bob.io.base"
#define BOB_IO_BASE_FULL_NAME "bob.io.base._library"

/* Maximum number of dimensions supported at this library */
#define BOB_BLITZ_MAXDIMS 4

#include <Python.h>
#include <bob.io.base/config.h>
#include <boost/shared_ptr.hpp>

/*******************
 * C API functions *
 *******************/

/* Enum defining entries in the function table */
enum _PyBobIo_ENUM{
  PyBobIo_APIVersion_NUM = 0,
  // C/C++ Type Information
  BobIoTypeinfo_Init_NUM,
  BobIoTypeinfo_Copy_NUM,
  BobIoTypeinfo_Set_NUM,
  BobIoTypeinfo_SetWithStrides_NUM,
  BobIoTypeinfo_SignedSet_NUM,
  BobIoTypeinfo_SignedSetWithStrides_NUM,
  BobIoTypeinfo_Reset_NUM,
  BobIoTypeinfo_IsValid_NUM,
  BobIoTypeinfo_HasValidShape_NUM,
  BobIoTypeinfo_ResetShape_NUM,
  BobIoTypeinfo_UpdateStrides_NUM,
  BobIoTypeinfo_Size_NUM,
  BobIoTypeinfo_BufferSize_NUM,
  BobIoTypeinfo_IsCompatible_NUM,
  BobIoTypeinfo_Str_NUM,
  // Data reordering
  BobIoReorder_RowToCol_NUM,
  BobIoReorder_ColToRow_NUM,
  BobIoReorder_RowToColComplex_NUM,
  BobIoReorder_ColToRowComplex_NUM,
  // Bindings for bob.io.base.File
  PyBobIoFile_Type_NUM,
  PyBobIoFileIterator_Type_NUM,
  // File loading and data type peeking
  BobIoFile_Open_NUM,
  BobIoFile_OpenWithExtension_NUM,
  BobIoFile_Peek_NUM,
  BobIoFile_PeekAll_NUM,
  // I/O generic bindings
  PyBobIo_TypeinfoAsTuple_NUM,
  PyBobIo_FilenameConverter_NUM,
  // HDF5 bindings
  PyBobIoHDF5File_Type_NUM,
  PyBobIoHDF5File_Check_NUM,
  PyBobIoHDF5File_Converter_NUM,
  // Codec registration and de-registration
  PyBobIoCodec_Register_NUM,
  PyBobIoCodec_Deregister_NUM,
  PyBobIoCodec_IsRegistered_NUM,
  PyBobIoCodec_GetDescription_NUM,
  // Total number of C API pointers
  PyBobIo_API_pointers
};

/**************
 * Versioning *
 **************/

#define PyBobIo_APIVersion_TYPE int

/********************
 * Type Information *
 ********************/

typedef struct {

    int dtype; ///< data type
    size_t nd; ///< number of dimensions
    size_t shape[BOB_BLITZ_MAXDIMS+1]; ///< length along each dimension
    size_t stride[BOB_BLITZ_MAXDIMS+1]; ///< strides along each dimension

} BobIoTypeinfo;


#define BobIoTypeinfo_Init_RET void
#define BobIoTypeinfo_Init_PROTO (BobIoTypeinfo*)

#define BobIoTypeinfo_Copy_RET int
#define BobIoTypeinfo_Copy_PROTO (BobIoTypeinfo*, const BobIoTypeinfo*)

#define BobIoTypeinfo_Set_RET int
#define BobIoTypeinfo_Set_PROTO (BobIoTypeinfo*, int, size_t, const size_t*)

#define BobIoTypeinfo_SetWithStrides_RET int
#define BobIoTypeinfo_SetWithStrides_PROTO (BobIoTypeinfo*, int, size_t, const size_t*, const size_t*)

#define BobIoTypeinfo_SignedSet_RET int
#define BobIoTypeinfo_SignedSet_PROTO (BobIoTypeinfo*, int, Py_ssize_t, const Py_ssize_t*)

#define BobIoTypeinfo_SignedSetWithStrides_RET int
#define BobIoTypeinfo_SignedSetWithStrides_PROTO (BobIoTypeinfo*, int, Py_ssize_t, const Py_ssize_t*, const Py_ssize_t*)

#define BobIoTypeinfo_Reset_RET void
#define BobIoTypeinfo_Reset_PROTO (BobIoTypeinfo*)

#define BobIoTypeinfo_IsValid_RET bool
#define BobIoTypeinfo_IsValid_PROTO (const BobIoTypeinfo*)

#define BobIoTypeinfo_HasValidShape_RET bool
#define BobIoTypeinfo_HasValidShape_PROTO (const BobIoTypeinfo*)

#define BobIoTypeinfo_ResetShape_RET void
#define BobIoTypeinfo_ResetShape_PROTO (BobIoTypeinfo*)

#define BobIoTypeinfo_UpdateStrides_RET int
#define BobIoTypeinfo_UpdateStrides_PROTO (BobIoTypeinfo*)

#define BobIoTypeinfo_Size_RET size_t
#define BobIoTypeinfo_Size_PROTO (const BobIoTypeinfo*)

#define BobIoTypeinfo_BufferSize_RET size_t
#define BobIoTypeinfo_BufferSize_PROTO (const BobIoTypeinfo*)

#define BobIoTypeinfo_IsCompatible_RET bool
#define BobIoTypeinfo_IsCompatible_PROTO (const BobIoTypeinfo*, const BobIoTypeinfo*)

#define BobIoTypeinfo_Str_RET std::string
#define BobIoTypeinfo_Str_PROTO (const BobIoTypeinfo*)

/********************
 * Array reordering *
 ********************/

#define BobIoReorder_RowToCol_RET int
#define BobIoReorder_RowToCol_PROTO (const void*, void*, const BobIoTypeinfo*)

#define BobIoReorder_ColToRow_RET int
#define BobIoReorder_ColToRow_PROTO (const void*, void*, const BobIoTypeinfo*)

#define BobIoReorder_RowToColComplex_RET int
#define BobIoReorder_RowToColComplex_PROTO (const void*, void*, const BobIoTypeinfo*)

#define BobIoReorder_ColToRowComplex_RET int
#define BobIoReorder_ColToRowComplex_PROTO (const void*, void*, const BobIoTypeinfo*)

/**********************************
 * Bindings for bob.io.base.File *
 **********************************/

/* Type definition for PyBobIoFileObject */

namespace bob { namespace io { namespace base { class File; }}}
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::base::File> f;

} PyBobIoFileObject;

#define PyBobIoFile_Type_TYPE PyTypeObject

typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  PyBobIoFileObject* pyfile;
  Py_ssize_t curpos;

} PyBobIoFileIteratorObject;

#define PyBobIoFileIterator_Type_TYPE PyTypeObject

/**************************************
 * File loading and data type peeking *
 **************************************/

#define BobIoFile_Open_RET boost::shared_ptr<bob::io::base::File>
#define BobIoFile_Open_PROTO (const char*, char)

#define BobIoFile_OpenWithExtension_RET boost::shared_ptr<bob::io::base::File>
#define BobIoFile_OpenWithExtension_PROTO (const char*, char, const char*)

#define BobIoFile_Peek_RET void
#define BobIoFile_Peek_PROTO (const char*, BobIoTypeinfo*)

#define BobIoFile_PeekAll_RET void
#define BobIoFile_PeekAll_PROTO (const char*, BobIoTypeinfo*)

/************************
 * I/O generic bindings *
 ************************/

#define PyBobIo_TypeinfoAsTuple_RET PyObject*
#define PyBobIo_TypeinfoAsTuple_PROTO (const BobIoTypeinfo& ti)

#define PyBobIo_FilenameConverter_RET int
#define PyBobIo_FilenameConverter_PROTO (PyObject* o, PyObject** b)

/*****************
 * HDF5 bindings *
 *****************/

namespace bob { namespace io { namespace base { class HDF5File; }}}
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  boost::shared_ptr<bob::io::base::HDF5File> f;

} PyBobIoHDF5FileObject;

#define PyBobIoHDF5File_Type_TYPE PyTypeObject

#define PyBobIoHDF5File_Check_RET int
#define PyBobIoHDF5File_Check_PROTO (PyObject* o)

#define PyBobIoHDF5File_Converter_RET int
#define PyBobIoHDF5File_Converter_PROTO (PyObject* o, PyBobIoHDF5FileObject** a)

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

/**
 * @brief This defines the factory method F that can create codecs. Your
 * task, as a codec developer is to create one of such methods for each of
 * your codecs and statically register them to the codec registry.
 *
 * Here are the meanings of the mode flag that should be respected by your
 * factory implementation:
 *
 * 'r': opens for reading only - no modifications can occur; it is an
 *      error to open a file that does not exist for read-only operations.
 * 'w': opens for reading and writing, but truncates the file if it
 *      exists; it is not an error to open files that do not exist with
 *      this flag.
 * 'a': opens for reading and writing - any type of modification can
 *      occur. If the file does not exist, this flag is effectively like
 *      'w'.
 *
 * Returns a newly allocated File object that can read and write data to the
 * file using a specific backend.
 */
typedef boost::shared_ptr<bob::io::base::File> (*BobIoFileFactory) (const char* filename, char mode);

#define PyBobIoCodec_Register_RET int
#define PyBobIoCodec_Register_PROTO (const char* extension, const char* description, BobIoFileFactory factory)

#define PyBobIoCodec_Deregister_RET int
#define PyBobIoCodec_Deregister_PROTO (const char* extension)

#define PyBobIoCodec_IsRegistered_RET int
#define PyBobIoCodec_IsRegistered_PROTO (const char* extension)

#define PyBobIoCodec_GetDescription_RET const char*
#define PyBobIoCodec_GetDescription_PROTO (const char* extension)

#ifdef BOB_IO_BASE_MODULE

  /* This section is used when compiling `bob.io.base' itself */

  /**************
   * Versioning *
   **************/

  extern int PyBobIo_APIVersion;

  /********************
   * Type Information *
   ********************/

  BobIoTypeinfo_Init_RET BobIoTypeinfo_Init BobIoTypeinfo_Init_PROTO;

  BobIoTypeinfo_Copy_RET BobIoTypeinfo_Copy BobIoTypeinfo_Copy_PROTO;

  BobIoTypeinfo_Set_RET BobIoTypeinfo_Set BobIoTypeinfo_Set_PROTO;

  BobIoTypeinfo_SetWithStrides_RET BobIoTypeinfo_SetWithStrides BobIoTypeinfo_SetWithStrides_PROTO;

  BobIoTypeinfo_SignedSet_RET BobIoTypeinfo_SignedSet BobIoTypeinfo_SignedSet_PROTO;

  BobIoTypeinfo_SignedSetWithStrides_RET BobIoTypeinfo_SignedSetWithStrides BobIoTypeinfo_SignedSetWithStrides_PROTO;

  BobIoTypeinfo_Reset_RET BobIoTypeinfo_Reset BobIoTypeinfo_Reset_PROTO;

  BobIoTypeinfo_IsValid_RET BobIoTypeinfo_IsValid BobIoTypeinfo_IsValid_PROTO;

  BobIoTypeinfo_HasValidShape_RET BobIoTypeinfo_HasValidShape BobIoTypeinfo_HasValidShape_PROTO;

  BobIoTypeinfo_ResetShape_RET BobIoTypeinfo_ResetShape BobIoTypeinfo_ResetShape_PROTO;

  BobIoTypeinfo_UpdateStrides_RET BobIoTypeinfo_UpdateStrides BobIoTypeinfo_UpdateStrides_PROTO;

  BobIoTypeinfo_Size_RET BobIoTypeinfo_Size BobIoTypeinfo_Size_PROTO;

  BobIoTypeinfo_BufferSize_RET BobIoTypeinfo_BufferSize BobIoTypeinfo_BufferSize_PROTO;

  BobIoTypeinfo_IsCompatible_RET BobIoTypeinfo_IsCompatible BobIoTypeinfo_IsCompatible_PROTO;

  BobIoTypeinfo_Str_RET BobIoTypeinfo_Str BobIoTypeinfo_Str_PROTO;

  /********************
   * Array reordering *
   ********************/

  BobIoReorder_RowToCol_RET BobIoReorder_RowToCol BobIoReorder_RowToCol_PROTO;

  BobIoReorder_ColToRow_RET BobIoReorder_ColToRow BobIoReorder_ColToRow_PROTO;

  BobIoReorder_RowToColComplex_RET BobIoReorder_RowToColComplex BobIoReorder_RowToColComplex_PROTO;

  BobIoReorder_ColToRowComplex_RET BobIoReorder_ColToRowComplex BobIoReorder_ColToRowComplex_PROTO;

  /**********************************
   * Bindings for bob.io.base.File *
   **********************************/

  extern PyBobIoFile_Type_TYPE PyBobIoFile_Type;
  extern PyBobIoFileIterator_Type_TYPE PyBobIoFileIterator_Type;

  /**************************************
   * File loading and data type peeking *
   **************************************/

  BobIoFile_Open_RET BobIoFile_Open BobIoFile_Open_PROTO;

  BobIoFile_OpenWithExtension_RET BobIoFile_OpenWithExtension BobIoFile_OpenWithExtension_PROTO;

  BobIoFile_Peek_RET BobIoFile_Peek BobIoFile_Peek_PROTO;

  BobIoFile_PeekAll_RET BobIoFile_PeekAll BobIoFile_PeekAll_PROTO;

  /************************
   * I/O generic bindings *
   ************************/

  PyBobIo_TypeinfoAsTuple_RET PyBobIo_TypeinfoAsTuple PyBobIo_TypeinfoAsTuple_PROTO;

  PyBobIo_FilenameConverter_RET PyBobIo_FilenameConverter PyBobIo_FilenameConverter_PROTO;

  /*****************
   * HDF5 bindings *
   *****************/

  extern PyBobIoHDF5File_Type_TYPE PyBobIoHDF5File_Type;

  PyBobIoHDF5File_Check_RET PyBobIoHDF5File_Check PyBobIoHDF5File_Check_PROTO;

  PyBobIoHDF5File_Converter_RET PyBobIoHDF5File_Converter PyBobIoHDF5File_Converter_PROTO;

  /*****************************************
   * Code Registration and De-registration *
   *****************************************/

  PyBobIoCodec_Register_RET PyBobIoCodec_Register PyBobIoCodec_Register_PROTO;

  PyBobIoCodec_Deregister_RET PyBobIoCodec_Deregister PyBobIoCodec_Deregister_PROTO;

  PyBobIoCodec_IsRegistered_RET PyBobIoCodec_IsRegistered PyBobIoCodec_IsRegistered_PROTO;

  PyBobIoCodec_GetDescription_RET PyBobIoCodec_GetDescription PyBobIoCodec_GetDescription_PROTO;

#else

  /* This section is used in modules that use `bob.io.base's' C-API */

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyBobIo_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyBobIo_API;
#    else
  static void **PyBobIo_API=NULL;
#    endif
#  endif

  /**************
   * Versioning *
   **************/

# define PyBobIo_APIVersion (*(PyBobIo_APIVersion_TYPE *)PyBobIo_API[PyBobIo_APIVersion_NUM])

  /********************
   * Type Information *
   ********************/

# define BobIoTypeinfo_Init (*(BobIoTypeinfo_Init_RET (*)BobIoTypeinfo_Init_PROTO) PyBobIo_API[BobIoTypeinfo_Init_NUM])

# define BobIoTypeinfo_Copy (*(BobIoTypeinfo_Copy_RET (*)BobIoTypeinfo_Copy_PROTO) PyBobIo_API[BobIoTypeinfo_Copy_NUM])

# define BobIoTypeinfo_Set (*(BobIoTypeinfo_Set_RET (*)BobIoTypeinfo_Set_PROTO) PyBobIo_API[BobIoTypeinfo_Set_NUM])

# define BobIoTypeinfo_SetWithStrides (*(BobIoTypeinfo_SetWithStrides_RET (*)BobIoTypeinfo_SetWithStrides_PROTO) PyBobIo_API[BobIoTypeinfo_SetWithStrides_NUM])

# define BobIoTypeinfo_SignedSet (*(BobIoTypeinfo_SignedSet_RET (*)BobIoTypeinfo_SignedSet_PROTO) PyBobIo_API[BobIoTypeinfo_SignedSet_NUM])

# define BobIoTypeinfo_SignedSetWithStrides (*(BobIoTypeinfo_SignedSetWithStrides_RET (*)BobIoTypeinfo_SignedSetWithStrides_PROTO) PyBobIo_API[BobIoTypeinfo_SignedSetWithStrides_NUM])

# define BobIoTypeinfo_Reset (*(BobIoTypeinfo_Reset_RET (*)BobIoTypeinfo_Reset_PROTO) PyBobIo_API[BobIoTypeinfo_Reset_NUM])

# define BobIoTypeinfo_IsValid (*(BobIoTypeinfo_IsValid_RET (*)BobIoTypeinfo_IsValid_PROTO) PyBobIo_API[BobIoTypeinfo_IsValid_NUM])

# define BobIoTypeinfo_HasValidShape (*(BobIoTypeinfo_HasValidShape_RET (*)BobIoTypeinfo_HasValidShape_PROTO) PyBobIo_API[BobIoTypeinfo_HasValidShape_NUM])

# define BobIoTypeinfo_ResetShape (*(BobIoTypeinfo_ResetShape_RET (*)BobIoTypeinfo_ResetShape_PROTO) PyBobIo_API[BobIoTypeinfo_ResetShape_NUM])

# define BobIoTypeinfo_UpdateStrides (*(BobIoTypeinfo_UpdateStrides_RET (*)BobIoTypeinfo_UpdateStrides_PROTO) PyBobIo_API[BobIoTypeinfo_UpdateStrides_NUM])

# define BobIoTypeinfo_Size (*(BobIoTypeinfo_Size_RET (*)BobIoTypeinfo_Size_PROTO) PyBobIo_API[BobIoTypeinfo_Size_NUM])

# define BobIoTypeinfo_BufferSize (*(BobIoTypeinfo_BufferSize_RET (*)BobIoTypeinfo_BufferSize_PROTO) PyBobIo_API[BobIoTypeinfo_BufferSize_NUM])

# define BobIoTypeinfo_IsCompatible (*(BobIoTypeinfo_IsCompatible_RET (*)BobIoTypeinfo_IsCompatible_PROTO) PyBobIo_API[BobIoTypeinfo_IsCompatible_NUM])

# define BobIoTypeinfo_Str (*(BobIoTypeinfo_Str_RET (*)BobIoTypeinfo_Str_PROTO) PyBobIo_API[BobIoTypeinfo_Str_NUM])

  /********************
   * Array reordering *
   ********************/

# define BobIoReorder_RowToCol (*(BobIoReorder_RowToCol_RET (*)BobIoReorder_RowToCol_PROTO) PyBobIo_API[BobIoReorder_RowToCol_NUM])

# define BobIoReorder_ColToRow (*(BobIoReorder_ColToRow_RET (*)BobIoReorder_ColToRow_PROTO) PyBobIo_API[BobIoReorder_ColToRow_NUM])

# define BobIoReorder_RowToColComplex (*(BobIoReorder_RowToColComplex_RET (*)BobIoReorder_RowToColComplex_PROTO) PyBobIo_API[BobIoReorder_RowToColComplex_NUM])

# define BobIoReorder_ColToRowComplex (*(BobIoReorder_ColToRowComplex_RET (*)BobIoReorder_ColToRowComplex_PROTO) PyBobIo_API[BobIoReorder_ColToRowComplex_NUM])

  /*****************************
   * Bindings for bob.io.File *
   *****************************/

# define PyBobIoFile_Type (*(PyBobIoFile_Type_TYPE *)PyBobIo_API[PyBobIoFile_Type_NUM])
# define PyBobIoFileIterator_Type (*(PyBobIoFileIterator_Type_TYPE *)PyBobIo_API[PyBobIoFileIterator_Type_NUM])

  /**************************************
   * File loading and data type peeking *
   **************************************/

# define BobIoFile_Open (*(BobIoFile_Open_RET (*)BobIoFile_Open_PROTO) PyBobIo_API[BobIoFile_Open_NUM])

# define BobIoFile_OpenWithExtension (*(BobIoFile_OpenWithExtension_RET (*)BobIoFile_OpenWithExtension_PROTO) PyBobIo_API[BobIoFile_OpenWithExtension_NUM])

# define BobIoFile_Peek (*(BobIoFile_Peek_RET (*)BobIoFile_Peek_PROTO) PyBobIo_API[BobIoFile_Peek_NUM])

# define BobIoFile_PeekAll (*(BobIoFile_PeekAll_RET (*)BobIoFile_PeekAll_PROTO) PyBobIo_API[BobIoFile_PeekAll_NUM])

  /************************
   * I/O generic bindings *
   ************************/

# define PyBobIo_TypeinfoAsTuple (*(PyBobIo_TypeinfoAsTuple_RET (*)PyBobIo_TypeinfoAsTuple_PROTO) PyBobIo_API[PyBobIo_TypeinfoAsTuple_NUM])

# define PyBobIo_FilenameConverter (*(PyBobIo_FilenameConverter_RET (*)PyBobIo_FilenameConverter_PROTO) PyBobIo_API[PyBobIo_FilenameConverter_NUM])

  /*****************
   * HDF5 bindings *
   *****************/

# define PyBobIoHDF5File_Type (*(PyBobIoHDF5File_Type_TYPE *)PyBobIo_API[PyBobIoHDF5File_Type_NUM])

# define PyBobIoHDF5File_Check (*(PyBobIoHDF5File_Check_RET (*)PyBobIoHDF5File_Check_PROTO) PyBobIo_API[PyBobIoHDF5File_Check_NUM])

# define PyBobIoHDF5File_Converter (*(PyBobIoHDF5File_Converter_RET (*)PyBobIoHDF5File_Converter_PROTO) PyBobIo_API[PyBobIoHDF5File_Converter_NUM])

/*****************************************
 * Code Registration and De-registration *
 *****************************************/

# define PyBobIoCodec_Register (*(PyBobIoCodec_Register_RET (*)PyBobIoCodec_Register_PROTO) PyBobIo_API[PyBobIoCodec_Register_NUM])

# define PyBobIoCodec_Deregister (*(PyBobIoCodec_Deregister_RET (*)PyBobIoCodec_Deregister_PROTO) PyBobIo_API[PyBobIoCodec_Deregister_NUM])

# define PyBobIoCodec_IsRegistered (*(PyBobIoCodec_IsRegistered_RET (*)PyBobIoCodec_IsRegistered_PROTO) PyBobIo_API[PyBobIoCodec_IsRegistered_NUM])

# define PyBobIoCodec_GetDescription (*(PyBobIoCodec_GetDescription_RET (*)PyBobIoCodec_GetDescription_PROTO) PyBobIo_API[PyBobIoCodec_GetDescription_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_bob_io_base(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(BOB_IO_BASE_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyBobIo_API = (void **)PyCapsule_GetPointer(c_api_object,
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyBobIo_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyBobIo_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" BOB_IO_BASE_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyBobIo_API[PyBobIo_APIVersion_NUM];

    if (BOB_IO_BASE_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, BOB_IO_BASE_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", BOB_IO_BASE_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* BOB_IO_BASE_MODULE */

#endif /* BOB_IO_BASE_H */
