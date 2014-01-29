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
#define XBOB_IO_MODULE
#include <xbob.io/config.h>

#include <string>
#include <cstdlib>
#include <boost/preprocessor/stringize.hpp>
#include <boost/format.hpp>

#include <bob/config.h>
#include <bob/io/CodecRegistry.h>
#include <bob/io/VideoUtilities.h>

#include <xbob.blitz/capi.h>
#include <xbob.blitz/cleanup.h>

extern "C" {

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <hdf5.h>
#include <jpeglib.h>

#define PNG_SKIP_SETJMP_CHECK
// #define requires because of the problematic pngconf.h.
// Look at the thread here:
// https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409
#include <png.h>

#if WITH_FFMPEG
#  include <libavformat/avformat.h>
#  include <libavcodec/avcodec.h>
#  include <libavutil/avutil.h>
#  include <libswscale/swscale.h>
#  include <libavutil/opt.h>
#  include <libavutil/pixdesc.h>
#  if !HAVE_FFMPEG_AVCODEC_AVCODECID
#    define AVCodecID CodecID
#  endif
#endif

#include <gif_lib.h>

#if WITH_MATIO
#include <matio.h>
#endif

#include <tiffio.h>

}

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

#if WITH_FFMPEG

static PyObject* make_object(bool v) {
  if (v) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject* make_object(unsigned int v) {
  return Py_BuildValue("n", v);
}

static PyObject* make_object(double v) {
  return PyFloat_FromDouble(v);
}

static PyObject* make_object(PyObject* v) {
  Py_INCREF(v);
  return v;
}

/**
 * Sets a dictionary entry using a string as key and another one as value.
 * Returns 1 in case of success, 0 in case of failure.
 */
template <typename T>
int dict_set_string(boost::shared_ptr<PyObject> d, const char* key, T value) {
  PyObject* pyvalue = make_object(value);
  if (!pyvalue) return 0;
  int retval = PyDict_SetItemString(d.get(), key, pyvalue);
  Py_DECREF(pyvalue);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

/**
 * Sets a dictionary entry using a string as key and another one as value.
 * Returns 1 in case of success, 0 in case of faiulre.
 */
template <typename T>
int list_append(PyObject* l, T value) {
  PyObject* pyvalue = make_object(value);
  if (!pyvalue) return 0;
  int retval = PyList_Append(l, pyvalue);
  Py_DECREF(pyvalue);
  if (retval == 0) return 1; //all good
  return 0; //a problem occurred
}

/**
 * A deleter, for shared_ptr's
 */
void pyobject_deleter(PyObject* o) {
  Py_XDECREF(o);
}

/**
 * Checks if it is a Python string for Python 2.x or 3.x
 */
int check_string(PyObject* o) {
#     if PY_VERSION_HEX >= 0x03000000
      return PyUnicode_Check(o);
#     else
      return PyString_Check(o);
#     endif
}

#endif /* WITH_FFMPEG */

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
 * FFmpeg version
 */
static PyObject* ffmpeg_version() {
  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

#if WITH_FFMPEG
# if defined(FFMPEG_VERSION)
  if (std::strlen(FFMPEG_VERSION)) {
    if (!dict_set(retval, "ffmpeg", FFMPEG_VERSION)) return 0;
  }
# endif
  if (!dict_set(retval, "avformat", BOOST_PP_STRINGIZE(LIBAVFORMAT_VERSION))) {
    return 0;
  }
  if (!dict_set(retval, "avcodec", BOOST_PP_STRINGIZE(LIBAVCODEC_VERSION))) {
    return 0;
  }
  if (!dict_set(retval, "avutil", BOOST_PP_STRINGIZE(LIBAVUTIL_VERSION))) {
    return 0;
  }
  if (!dict_set(retval, "swscale", BOOST_PP_STRINGIZE(LIBSWSCALE_VERSION))) {
    return 0;
  }
#else
  if (!dict_set(retval, "ffmpeg", "unavailable")) {
    return 0;
  }
#endif
  Py_INCREF(retval);
  return retval;
}

/**
 * LibJPEG version
 */
static PyObject* libjpeg_version() {
  boost::format f("%d (compiled with %d bits depth)");
  f % JPEG_LIB_VERSION;
  f % BITS_IN_JSAMPLE;
  return Py_BuildValue("s", f.str().c_str());
}

/**
 * Libpng version
 */
static PyObject* libpng_version() {
  return Py_BuildValue("s", PNG_LIBPNG_VER_STRING);
}

/**
 * Libtiff version
 */
static PyObject* libtiff_version() {

  static const std::string beg_str("LIBTIFF, Version ");
  static const size_t beg_len = beg_str.size();
  std::string vtiff(TIFFGetVersion());

  // Remove first part if it starts with "LIBTIFF, Version "
  if(vtiff.compare(0, beg_len, beg_str) == 0)
    vtiff = vtiff.substr(beg_len);

  // Remove multiple (copyright) lines if any
  size_t end_line = vtiff.find("\n");
  if(end_line != std::string::npos)
    vtiff = vtiff.substr(0,end_line);

  return Py_BuildValue("s", vtiff.c_str());

}

/**
 * Version of giflib support
 */
static PyObject* giflib_version() {
#ifdef GIF_LIB_VERSION
 return Py_BuildValue("s", GIF_LIB_VERSION);
#else
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(GIFLIB_MAJOR);
  f % BOOST_PP_STRINGIZE(GIFLIB_MINOR);
  f % BOOST_PP_STRINGIZE(GIFLIB_RELEASE);
  return Py_BuildValue("s", f.str().c_str());
#endif
}


/**
 * Matio, if compiled with such support
 */
static PyObject* matio_version() {
#if WITH_MATIO
  boost::format f("%s.%s.%s");
  f % BOOST_PP_STRINGIZE(MATIO_MAJOR_VERSION);
  f % BOOST_PP_STRINGIZE(MATIO_MINOR_VERSION);
  f % BOOST_PP_STRINGIZE(MATIO_RELEASE_LEVEL);
  return Py_BuildValue("s", f.str().c_str());
#else
  return Py_BuildValue("s", "unavailable");
#endif
}

static PyObject* build_version_dictionary() {

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  if (!dict_steal(retval, "HDF5", hdf5_version())) return 0;

  if (!dict_steal(retval, "FFmpeg", ffmpeg_version())) return 0;

  if (!dict_steal(retval, "libjpeg", libjpeg_version())) return 0;

  if (!dict_set(retval, "libnetpbm", "Unknown version")) return 0;

  if (!dict_steal(retval, "libpng", libpng_version())) return 0;

  if (!dict_steal(retval, "libtiff", libtiff_version())) return 0;

  if (!dict_steal(retval, "giflib", giflib_version())) return 0;

  if (!dict_steal(retval, "MatIO", matio_version())) return 0;

  Py_INCREF(retval);
  return retval;
}

/***********************************************************
 * FFmpeg information
 ***********************************************************/

#if WITH_FFMPEG

/**
 * Describes a given codec. We return a **new reference** to a dictionary
 * containing the codec properties.
 */
static PyObject* describe_codec(const AVCodec* codec) {

  /**
   * We wrap the returned object into a smart pointer until we
   * are absolutely sure all went good. At this point, we free
   * the PyObject* from the boost encapsulation and return it.
   */
  boost::shared_ptr<PyObject> retval(PyDict_New(), &pyobject_deleter);
  if (!retval) return 0;

  /* Sets basic properties for the codec */
  if (!dict_set_string(retval, "name", codec->name)) return 0;
  if (!dict_set_string(retval, "long_name", codec->long_name)) return 0;
  if (!dict_set_string(retval, "id", (unsigned int)codec->id)) return 0;

  /**
   * If pixel formats are available, creates and attaches a
   * list with all their names
   */

  boost::shared_ptr<PyObject> pixfmt;
  if (codec->pix_fmts) {
    pixfmt.reset(PyList_New(0), &pyobject_deleter);
    if (!pixfmt) return 0;

    unsigned int i=0;
    while(codec->pix_fmts[i] != -1) {
      if (!list_append(pixfmt.get(),
#if LIBAVUTIL_VERSION_INT >= 0x320f01 //50.15.1 @ ffmpeg-0.6
            av_get_pix_fmt_name
#else
            avcodec_get_pix_fmt_name
#endif
            (codec->pix_fmts[i++]))) return 0;
    }
    pixfmt.reset(PySequence_Tuple(pixfmt.get()), &pyobject_deleter);
  }
  else {
    Py_INCREF(Py_None);
    pixfmt.reset(Py_None, &pyobject_deleter);
  }

  if (!dict_set_string(retval, "pixfmts", pixfmt.get())) return 0;

  /* Get specific framerates for the codec, if any */
  const AVRational* rate = codec->supported_framerates;
  boost::shared_ptr<PyObject> rates(PyList_New(0), &pyobject_deleter);
  if (!rates) return 0;

  while (rate && rate->num && rate->den) {
    list_append(rates.get(), ((double)rate->num)/((double)rate->den));
    ++rate;
  }
  rates.reset(PySequence_Tuple(rates.get()), &pyobject_deleter);
  if (!rates) return 0;

  if (!dict_set_string(retval, "specific_framerates_hz", rates.get())) return 0;

  /* Other codec capabilities */
# ifdef CODEC_CAP_LOSSLESS
  if (!dict_set_string(retval, "lossless", (bool)(codec->capabilities & CODEC_CAP_LOSSLESS))) return 0;
# endif
# ifdef CODEC_CAP_EXPERIMENTAL
  if (!dict_set_string(retval, "experimental", (bool)(codec->capabilities & CODEC_CAP_EXPERIMENTAL))) return 0;
# endif
# ifdef CODEC_CAP_DELAY
  if (!dict_set_string(retval, "delay", (bool)(codec->capabilities & CODEC_CAP_DELAY))) return 0;
# endif
# ifdef CODEC_CAP_HWACCEL
  if (!dict_set_string(retval, "hardware_accelerated", (bool)(codec->capabilities & CODEC_CAP_HWACCEL))) return 0;
# endif
  if (!dict_set_string(retval, "encode", (bool)(avcodec_find_encoder(codec->id)))) return 0;
  if (!dict_set_string(retval, "decode", (bool)(avcodec_find_decoder(codec->id)))) return 0;

  /* If all went OK, detach the returned value from the smart pointer **/
  Py_INCREF(retval.get());
  return retval.get();

}

/**
 * Describes a given codec or raises, in case the codec cannot be accessed
 */
static PyObject* PyBobIo_DescribeEncoder(PyObject*, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &key)) return 0;

  if (!PyNumber_Check(key) && !check_string(key)) {
    PyErr_SetString(PyExc_TypeError, "input `key' must be a number identifier or a string with the codec name");
    return 0;
  }

  if (PyNumber_Check(key)) {

    /* If you get to this point, the user passed a number - re-parse */
    int id = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &id)) return 0;

    AVCodec* codec = avcodec_find_encoder((AVCodecID)id);
    if (!codec) {
      PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_encoder(%d == 0x%x) did not return a valid codec", id, id);
      return 0;
    }

    return describe_codec(codec);
  }

  /* If you get to this point, the user passed a string - re-parse */
  const char* name = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &name)) return 0;

  AVCodec* codec = avcodec_find_encoder_by_name(name);
  if (!codec) {
    PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_encoder_by_name(`%s') did not return a valid codec", name);
    return 0;
  }

  return describe_codec(codec);
}

PyDoc_STRVAR(s_describe_encoder_str, "describe_encoder");
PyDoc_STRVAR(s_describe_encoder_doc,
"describe_encoder([key]) -> dict\n\
\n\
Parameters:\n\
\n\
key\n\
  [int|str, optional] A key which can be either the encoder\n\
  name or its numerical identifier.\n\
\n\
Returns a dictionary containing a description of properties in\n\
the given encoder.\n\
");

/**
 * Describes a given codec or raises, in case the codec cannot be accessed
 */
static PyObject* PyBobIo_DescribeDecoder(PyObject*, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"key", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* key = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &key)) return 0;

  if (!PyNumber_Check(key) && !check_string(key)) {
    PyErr_SetString(PyExc_TypeError, "input `key' must be a number identifier or a string with the codec name");
    return 0;
  }

  if (PyNumber_Check(key)) {

    /* If you get to this point, the user passed a number - re-parse */
    int id = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &id)) return 0;

    AVCodec* codec = avcodec_find_decoder((AVCodecID)id);
    if (!codec) {
      PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_decoder(%d == 0x%x) did not return a valid codec", id, id);
      return 0;
    }

    return describe_codec(codec);
  }

  /* If you get to this point, the user passed a string - re-parse */
  const char* name = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &name)) return 0;

  AVCodec* codec = avcodec_find_decoder_by_name(name);
  if (!codec) {
    PyErr_Format(PyExc_RuntimeError, "ffmpeg::avcodec_find_decoder_by_name(`%s') did not return a valid codec", name);
    return 0;
  }

  return describe_codec(codec);
}

PyDoc_STRVAR(s_describe_decoder_str, "describe_decoder");
PyDoc_STRVAR(s_describe_decoder_doc,
"describe_decoder([key]) -> dict\n\
\n\
Parameters:\n\
\n\
key\n\
  [int|str, optional] A key which can be either the decoder\n\
  name or its numerical identifier.\n\
\n\
Returns a dictionary containing a description of properties in\n\
the given decoder.\n\
");

static PyObject* get_video_codecs(void (*f)(std::map<std::string, const AVCodec*>&)) {

  std::map<std::string, const AVCodec*> m;
  f(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {
    PyObject* pyvalue = describe_codec(k->second);
    if (!pyvalue) return 0;
    auto pyvalue_ = make_safe(pyvalue);
    if (PyDict_SetItemString(retval, k->first.c_str(), pyvalue) != 0) return 0;
  }

  Py_INCREF(retval);
  return retval;

}

static PyObject* PyBobIo_SupportedCodecs(PyObject*) {
  return get_video_codecs(&bob::io::detail::ffmpeg::codecs_supported);
}

static PyObject* PyBobIo_AvailableCodecs(PyObject*) {
  return get_video_codecs(&bob::io::detail::ffmpeg::codecs_installed);
}

PyDoc_STRVAR(s_supported_codecs_str, "supported_video_codecs");
PyDoc_STRVAR(s_supported_codecs_doc,
"supported_video_codecs() -> dict\n\
\n\
Returns a dictionary with currently supported video codec properties.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in codecs for videos that are fully supported.\n\
");

PyDoc_STRVAR(s_available_codecs_str, "available_video_codecs");
PyDoc_STRVAR(s_available_codecs_doc,
"available_video_codecs() -> dict\n\
\n\
Returns a dictionary with all available video codec properties.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in codecs for videos that are available but **not necessarily\n\
supported**.\n\
");

static PyObject* get_video_iformats(void (*f)(std::map<std::string, AVInputFormat*>&)) {

  std::map<std::string, AVInputFormat*> m;
  f(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {

    PyObject* property = PyDict_New();
    if (!property) return 0;
    auto property_ = make_safe(property);

    if (!dict_set(property, "name", k->second->name)) return 0;

    if (!dict_set(property, "long_name", k->second->long_name)) return 0;

    // get extensions
    std::vector<std::string> exts;
    bob::io::detail::ffmpeg::tokenize_csv(k->second->extensions, exts);

    PyObject* ext_list = PyList_New(0);
    if (!ext_list) return 0;
    auto ext_list_ = make_safe(ext_list);

    for (auto ext=exts.begin(); ext!=exts.end(); ++ext) {
      if (!list_append(ext_list, ext->c_str())) return 0;
    }

    Py_INCREF(ext_list);
    if (!dict_steal(property, "extensions", ext_list)) return 0;

    Py_INCREF(property);
    if (!dict_steal(retval, k->first.c_str(), property)) return 0;

  }

  Py_INCREF(retval);
  return retval;

}

static PyObject* PyBobIo_SupportedInputFormats(PyObject*) {
  return get_video_iformats(&bob::io::detail::ffmpeg::iformats_supported);
}

static PyObject* PyBobIo_AvailableInputFormats(PyObject*) {
  return get_video_iformats(&bob::io::detail::ffmpeg::iformats_installed);
}

PyDoc_STRVAR(s_supported_iformats_str, "supported_videoreader_formats");
PyDoc_STRVAR(s_supported_iformats_doc,
"supported_videoreader_formats() -> dict\n\
\n\
Returns a dictionary with currently supported video input formats.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in input formats for videos that are fully supported.\n\
");

PyDoc_STRVAR(s_available_iformats_str, "available_videoreader_formats");
PyDoc_STRVAR(s_available_iformats_doc,
"available_videoreader_formats() -> dict\n\
\n\
Returns a dictionary with currently available video input formats.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in input formats for videos that are available, but **not\n\
necessarily supported** by this library.\n\
");

static PyObject* get_video_oformats(bool supported) {

  std::map<std::string, AVOutputFormat*> m;
  if (supported) bob::io::detail::ffmpeg::oformats_supported(m);
  else bob::io::detail::ffmpeg::oformats_installed(m);

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto k=m.begin(); k!=m.end(); ++k) {

    PyObject* property = PyDict_New();
    if (!property) return 0;
    auto property_ = make_safe(property);

    if (!dict_set(property, "name", k->second->name)) return 0;

    if (!dict_set(property, "long_name", k->second->long_name)) return 0;

    if (!dict_set(property, "mime_type", k->second->mime_type)) return 0;

    // get extensions
    std::vector<std::string> exts;
    bob::io::detail::ffmpeg::tokenize_csv(k->second->extensions, exts);

    PyObject* ext_list = PyList_New(0);
    if (!ext_list) return 0;
    auto ext_list_ = make_safe(ext_list);

    for (auto ext=exts.begin(); ext!=exts.end(); ++ext) {
      if (!list_append(ext_list, ext->c_str())) return 0;
    }

    Py_INCREF(ext_list);
    if (!dict_steal(property, "extensions", ext_list)) return 0;

    /**  get recommended codec **/
    PyObject* default_codec = 0;
    if (k->second->video_codec) {
      AVCodec* codec = avcodec_find_encoder(k->second->video_codec);
      if (codec) {
        default_codec = describe_codec(codec);
        if (!default_codec) return 0;
      }
    }

    if (!default_codec) {
      Py_INCREF(Py_None);
      default_codec = Py_None;
    }

    if (!dict_steal(property, "default_codec", default_codec)) return 0;

    /** get supported codec list **/
    if (supported) {
      std::vector<const AVCodec*> codecs;
      bob::io::detail::ffmpeg::oformat_supported_codecs(k->second->name, codecs);

      PyObject* supported_codecs = PyDict_New();
      if (!supported_codecs) return 0;
      auto supported_codecs_ = make_safe(supported_codecs);

      for (auto c=codecs.begin(); c!=codecs.end(); ++c) {
        PyObject* codec_descr = describe_codec(*c);
        auto codec_descr_ = make_safe(codec_descr);
        if (!codec_descr) return 0;
        Py_INCREF(codec_descr);
        if (!dict_steal(supported_codecs, (*c)->name, codec_descr)) return 0;
      }

      Py_INCREF(supported_codecs);
      if (!dict_steal(property, "supported_codecs", supported_codecs)) return 0;
    }

    Py_INCREF(property);
    if (!dict_steal(retval, k->first.c_str(), property)) return 0;

  }

  Py_INCREF(retval);
  return retval;

}

static PyObject* PyBobIo_SupportedOutputFormats(PyObject*) {
  return get_video_oformats(true);
}

static PyObject* PyBobIo_AvailableOutputFormats(PyObject*) {
  return get_video_oformats(false);
}

PyDoc_STRVAR(s_supported_oformats_str, "supported_videowriter_formats");
PyDoc_STRVAR(s_supported_oformats_doc,
"supported_videowriter_formats() -> dict\n\
\n\
Returns a dictionary with currently supported video output formats.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in output formats for videos that are fully supported.\n\
");

PyDoc_STRVAR(s_available_oformats_str, "available_videowriter_formats");
PyDoc_STRVAR(s_available_oformats_doc,
"available_videowriter_formats() -> dict\n\
\n\
Returns a dictionary with currently available video output formats.\n\
\n\
Returns a dictionary containing a detailed description of the\n\
built-in output formats for videos that are available, but **not\n\
necessarily supported** by this library.\n\
");

#endif /* WITH_FFMPEG */

static PyObject* PyBobIo_Extensions(PyObject*) {

  typedef std::map<std::string, std::string> map_type;
  const map_type& table = bob::io::CodecRegistry::getExtensions();

  PyObject* retval = PyDict_New();
  if (!retval) return 0;
  auto retval_ = make_safe(retval);

  for (auto it=table.begin(); it!=table.end(); ++it) {
    PyObject* pyvalue = make_object(it->second.c_str());
    if (!pyvalue) return 0;
    auto pyvalue_ = make_safe(retval);
    if (PyDict_SetItemString(retval, it->first.c_str(), pyvalue) != 0) return 0;
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

#if WITH_FFMPEG
    {
      s_describe_encoder_str,
      (PyCFunction)PyBobIo_DescribeEncoder,
      METH_VARARGS|METH_KEYWORDS,
      s_describe_encoder_doc,
    },
    {
      s_describe_decoder_str,
      (PyCFunction)PyBobIo_DescribeDecoder,
      METH_VARARGS|METH_KEYWORDS,
      s_describe_decoder_doc,
    },
    {
      s_supported_codecs_str,
      (PyCFunction)PyBobIo_SupportedCodecs,
      METH_NOARGS,
      s_supported_codecs_doc,
    },
    {
      s_available_codecs_str,
      (PyCFunction)PyBobIo_AvailableCodecs,
      METH_NOARGS,
      s_available_codecs_doc,
    },
    {
      s_supported_iformats_str,
      (PyCFunction)PyBobIo_SupportedInputFormats,
      METH_NOARGS,
      s_supported_iformats_doc,
    },
    {
      s_available_iformats_str,
      (PyCFunction)PyBobIo_AvailableInputFormats,
      METH_NOARGS,
      s_available_iformats_doc,
    },
    {
      s_supported_oformats_str,
      (PyCFunction)PyBobIo_SupportedOutputFormats,
      METH_NOARGS,
      s_supported_oformats_doc,
    },
    {
      s_available_oformats_str,
      (PyCFunction)PyBobIo_AvailableOutputFormats,
      METH_NOARGS,
      s_available_oformats_doc,
    },
#endif /* WITH_FFMPEG */

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
  if (PyModule_AddIntConstant(m, "__api_version__", XBOB_IO_API_VERSION) < 0) 
    return 0;
  if (PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION) < 0)
    return 0;
  if (PyModule_AddObject(m, "versions", build_version_dictionary()) < 0) return 0;

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
