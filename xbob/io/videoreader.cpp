/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  6 Nov 21:44:34 2013
 *
 * @brief Bindings to bob::io::VideoReader
 */

#define XBOB_IO_MODULE
#include <xbob.io/api.h>

#if WITH_FFMPEG
#include <boost/make_shared.hpp>

#define VIDEOREADER_NAME VideoReader
PyDoc_STRVAR(s_videoreader_str, BOOST_PP_STRINGIZE(XBOB_IO_MODULE_PREFIX) "." BOOST_PP_STRINGIZE(VIDEOREADER_NAME));

/* How to create a new PyBobIoVideoReaderObject */
static PyObject* PyBobIoVideoReader_New(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobIoVideoReaderObject* self = (PyBobIoVideoReaderObject*)type->tp_alloc(type, 0);

  self->v.reset();

  return reinterpret_cast<PyObject*>(self);
}

static void PyBobIoVideoReader_Delete (PyBobIoVideoReaderObject* o) {

  o->v.reset();
  o->ob_type->tp_free((PyObject*)o);

}

/* The __init__(self) method */
static int PyBobIoVideoReader_Init(PyBobIoVideoReaderObject* self, 
    PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"filename", "check", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  char* filename = 0;
  PyObject* pycheck = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|O", kwlist, 
        &filename, &pycheck)) return -1;

  if (pycheck && !PyBool_Check(pycheck)) {
    PyErr_SetString(PyExc_TypeError, "argument to `check' must be a boolean");
    return -1;
  }

  bool check = false;
  if (pycheck && (pycheck == Py_True)) check = true;

  try {
    self->v = boost::make_shared<bob::io::VideoReader>(filename, check);
  }
  catch (std::exception& e) {
    PyErr_Format(PyExc_RuntimeError, "cannot open video file `%s' for reading: %s", filename, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot open video file `%s' for reading: unknown exception caught", filename);
    return -1;
  }

  return 0; ///< SUCCESS
}

PyDoc_STRVAR(s_videoreader_doc,
"VideoReader(filename, [check=True]) -> new bob::io::VideoReader\n\
\n\
Use this object to read frames from video files.\n\
\n\
Constructor parameters:\n\
\n\
filename\n\
  [str] The file path to the file you want to read data from\n\
\n\
check\n\
  [bool] Format and codec will be extracted from the video metadata.\n\
  By default, if the format and/or the codec are not\n\
  supported by this version of Bob, an exception will be raised.\n\
  You can (at your own risk) set this flag to ``False`` to\n\
  avoid this check.\n\
\n\
VideoReader objects can read data from video files. The current\n\
implementation uses `FFmpeg <http://ffmpeg.org>`_ (or\n\
`libav <http://libav.org>`_ if FFmpeg is not available) which is\n\
a stable freely available video encoding and decoding library,\n\
designed specifically for these tasks. You can read an entire\n\
video in memory by using the 'load()' method or use iterators\n\
to read it frame by frame and avoid overloading your machine's\n\
memory. The maximum precision data `FFmpeg` will yield is a 24-bit\n\
(8-bit per band) representation of each pixel (32-bit depths are\n\
also supported by `FFmpeg`, but not by this extension presently).\n\
So, the output of data is done with ``uint8`` as data type.\n\
Output will be colored using the RGB standard, with each band\n\
varying between 0 and 255, with zero meaning pure black and 255,\n\
pure white (color).\n\
");

PyTypeObject PyBobIoVideoReader_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_videoreader_str,                          /*tp_name*/
    sizeof(PyBobIoVideoReaderObject),           /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobIoVideoReader_Delete,      /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    0, //(reprfunc)PyBobIoVideoReader_Repr,                 /*tp_repr*/
    0,                                          /*tp_as_number*/
    0, //&PyBobIoVideoReader_Sequence,                      /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    0, //(reprfunc)PyBobIoVideoReader_Repr,                 /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_videoreader_doc,                          /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,                                          /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    0, //PyBobIoVideoReader_Methods,                        /* tp_methods */
    0,                                          /* tp_members */
    0, //PyBobIoVideoReader_getseters,                      /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobIoVideoReader_Init,                 /* tp_init */
    0,                                          /* tp_alloc */
    PyBobIoVideoReader_New,                            /* tp_new */
};
/**
  
    .add_property("filename", make_function(&bob::io::VideoReader::filename, return_value_policy<copy_const_reference>()), "The full path to the file that will be decoded by this object")

    .add_property("height", &bob::io::VideoReader::height, "The height of each frame in the video (a multiple of 2)")

    .add_property("width", &bob::io::VideoReader::width, "The width of each frame in the video (a multiple of 2)")

    .add_property("number_of_frames", &bob::io::VideoReader::numberOfFrames, "The number of frames in this video file")

    .def("__len__", &bob::io::VideoReader::numberOfFrames)

    .add_property("duration", &bob::io::VideoReader::duration, "Total duration of this video file in microseconds (long)")

    .add_property("format_name", make_function(&bob::io::VideoReader::formatName, return_value_policy<copy_const_reference>()), "Short name of the format in which this video file was recorded in")

    .add_property("format_long_name", make_function(&bob::io::VideoReader::formatLongName, return_value_policy<copy_const_reference>()), "Verbose name of the format in which this video file was recorded in")

    .add_property("codec_name", make_function(&bob::io::VideoReader::codecName, return_value_policy<copy_const_reference>()), "Short name of the codec that will be used to decode this video file")

    .add_property("codec_long_name", make_function(&bob::io::VideoReader::codecLongName, return_value_policy<copy_const_reference>()), "Verbose name of the codec that will be used to decode this video file")

    .add_property("frame_rate", &bob::io::VideoReader::frameRate, "Video's announced frame rate (note there are video formats with variable frame rates)")

    .add_property("info", make_function(&bob::io::VideoReader::info, return_value_policy<copy_const_reference>()), "Informative string containing many details of this video and available ffmpeg bindings that will read it")

    .add_property("video_type", make_function(&bob::io::VideoReader::video_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")

    .add_property("frame_type", make_function(&bob::io::VideoReader::frame_type, return_value_policy<copy_const_reference>()), "Typing information to load the file frame by frame.")

    .def("__load__", &videoreader_load, videoreader_load_overloads((arg("self"), arg("raise_on_error")=false), "Loads all of the video stream in a numpy ndarray organized in this way: (frames, color-bands, height, width). I'll dynamically allocate the output array and return it to you. The flag ``raise_on_error``, which is set to ``False`` by default influences the error reporting in case problems are found with the video file. If you set it to ``True``, we will report problems raising exceptions. If you either don't set it or set it to ``False``, we will truncate the file at the frame with problems and will not report anything. It is your task to verify if the number of frames returned matches the expected number of frames as reported by the property ``number_of_frames`` in this object."))

    .def("__iter__", &bob::io::VideoReader::begin, with_custodian_and_ward_postcall<0,1>())

    .def("__getitem__", &videoreader_getitem)

    .def("__getitem__", &videoreader_getslice)

**/

#endif /* WITH_FFMPEG */
