#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed Jun 22 17:50:08 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Runs some video tests
"""

import os
import sys
import numpy
import nose.tools
from . import utils as testutils
from ..utils import color_distortion, frameskip_detection, quality_degradation

# These are some global parameters for the test.
INPUT_VIDEO = testutils.datafile('test.mov', __name__)

@testutils.ffmpeg_found()
def test_codec_support():

  # Describes all encoders
  from .._externals import describe_encoder, describe_decoder, supported_video_codecs

  supported = supported_video_codecs()

  for k,v in supported.items():
    # note: searching by name (using `k') will not always work
    if v['decode']: assert describe_decoder(v['id'])
    if v['encode']: assert describe_encoder(v['id'])

  # Assert we support, at least, some known codecs
  for codec in ('ffv1', 'zlib', 'wmv2', 'mpeg4', 'mjpeg'):
    assert codec in supported
    assert supported[codec]['encode']
    assert supported[codec]['decode']

@testutils.ffmpeg_found()
def test_input_format_support():

  # Describes all encoders
  from .._externals import supported_videoreader_formats

  supported = supported_videoreader_formats()

  # Assert we support, at least, some known codecs
  for fmt in ('avi', 'mov', 'mp4'):
    assert fmt in supported

@testutils.ffmpeg_found()
def test_output_format_support():

  # Describes all encoders
  from .._externals import supported_videowriter_formats

  supported = supported_videowriter_formats()

  # Assert we support, at least, some known codecs
  for fmt in ('avi', 'mov', 'mp4'):
    assert fmt in supported

@testutils.ffmpeg_found()
def test_video_reader_attributes():

  from .. import VideoReader

  iv = VideoReader(INPUT_VIDEO)

  assert isinstance(iv.filename, str)
  assert isinstance(iv.height, int)
  assert isinstance(iv.width, int)
  assert iv.height != iv.width
  assert isinstance(iv.duration, int)
  assert isinstance(iv.format_name, str)
  assert isinstance(iv.format_long_name, str)
  assert isinstance(iv.codec_name, str)
  assert isinstance(iv.codec_long_name, str)
  assert isinstance(iv.frame_rate, float)
  assert isinstance(iv.video_type, tuple)
  assert len(iv.video_type) == 3
  assert isinstance(iv.video_type[0], numpy.dtype)
  assert isinstance(iv.video_type[1], tuple)
  assert isinstance(iv.video_type[2], tuple)
  assert isinstance(iv.frame_type, tuple)
  assert len(iv.frame_type) == 3
  assert iv.frame_type[0] == iv.video_type[0]
  assert isinstance(iv.video_type[1], tuple)
  nose.tools.eq_(len(iv.video_type[1]), len(iv.frame_type[1])+1)
  nose.tools.eq_(len(iv.video_type[2]), len(iv.frame_type[2])+1)
  assert isinstance(iv.info, str)

@testutils.ffmpeg_found()
def test_video_reader_str():

  from .. import VideoReader

  iv = VideoReader(INPUT_VIDEO)
  assert repr(iv)
  assert str(iv)

@testutils.ffmpeg_found()
def test_indexing():

  from .. import VideoReader
  f = VideoReader(INPUT_VIDEO)

  nose.tools.eq_(len(f), 375)

  objs = f[:10]
  nose.tools.eq_(len(objs), 10)
  obj0 = f[0]
  obj1 = f[1]

  # simple indexing
  assert numpy.allclose(objs[0], obj0)
  assert numpy.allclose(objs[1], obj1)
  assert numpy.allclose(f[len(f)-1], f[-1])
  assert numpy.allclose(f[len(f)-2], f[-2])

@testutils.ffmpeg_found()
def test_slicing_0():

  from .. import load, VideoReader
  f = VideoReader(INPUT_VIDEO)

  objs = f[:]
  for i, k in enumerate(load(INPUT_VIDEO)):
    assert numpy.allclose(k, objs[i])

@testutils.ffmpeg_found()
def test_slicing_1():

  from .. import VideoReader
  f = VideoReader(INPUT_VIDEO)

  s = f[3:10:2]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[3])
  assert numpy.allclose(s[1], f[5])
  assert numpy.allclose(s[2], f[7])
  assert numpy.allclose(s[3], f[9])

@testutils.ffmpeg_found()
def test_slicing_2():

  from .. import VideoReader
  f = VideoReader(INPUT_VIDEO)

  s = f[-10:-2:3]
  nose.tools.eq_(len(s), 3)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-7])
  assert numpy.allclose(s[2], f[len(f)-4])

@testutils.ffmpeg_found()
def test_slicing_3():

  from .. import VideoReader
  f = VideoReader(INPUT_VIDEO)
  objs = f.load()

  # get negative stepping slice
  s = f[20:10:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[20])
  assert numpy.allclose(s[1], f[17])
  assert numpy.allclose(s[2], f[14])
  assert numpy.allclose(s[3], f[11])

@testutils.ffmpeg_found()
def test_slicing_4():

  from .. import VideoReader
  f = VideoReader(INPUT_VIDEO)
  objs = f[:21]

  # get all negative slice
  s = f[-10:-20:-3]
  nose.tools.eq_(len(s), 4)
  assert numpy.allclose(s[0], f[len(f)-10])
  assert numpy.allclose(s[1], f[len(f)-13])
  assert numpy.allclose(s[2], f[len(f)-16])
  assert numpy.allclose(s[3], f[len(f)-19])


@testutils.ffmpeg_found()
def test_can_use_array_interface():

  from .. import load, VideoReader
  array = load(INPUT_VIDEO)
  iv = VideoReader(INPUT_VIDEO)

  for frame_id, frame in zip(range(array.shape[0]), iv.__iter__()):
    assert numpy.array_equal(array[frame_id,:,:,:], frame)

@testutils.ffmpeg_found()
def test_can_iterate():

  # This test shows how you can read image frames from a VideoReader created
  # on the spot
  from .. import VideoReader
  video = VideoReader(INPUT_VIDEO)
  counter = 0
  for frame in video:
    assert isinstance(frame, numpy.ndarray)
    assert len(frame.shape) == 3
    assert frame.shape[0] == 3 #color-bands (RGB)
    assert frame.shape[1] == 240 #height
    assert frame.shape[2] == 320 #width
    counter += 1

  assert counter == len(video) #we have gone through all frames

@testutils.ffmpeg_found()
def check_format_codec(function, shape, framerate, format, codec, maxdist):

  length, height, width = shape
  fname = testutils.temporary_filename(suffix='.%s' % format)

  try:
    orig, framerate, encoded = function(shape, framerate, format, codec, fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded" % (len(orig), len(encoded))
    assert len(orig) == len(reloaded), "original length %d != %d reloaded" % (len(orig), len(reloaded))

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed" % (max(dist), maxdist)

    # assert we can randomly access any frame (choose 3 at random)
    for k in numpy.random.randint(length, size=(3,)):
      rdist = abs(orig[k].astype('float64')-encoded[k].astype('float64')).mean()
      assert rdist <= maxdist, "distortion(frame[%d]) %g > %g allowed" % (k, rdist, maxdist)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(framerate - encoded.frame_rate) <= (1.0/length), "reloaded framerate %g differs from original %g by more than %g" % (encoded.frame_rate, framerate, 1.0/length)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_format_codecs():

  length = 30
  width = 128
  height = 128
  framerate = 30.
  shape = (length, height, width)
  methods = dict(
      frameskip = frameskip_detection,
      color     = color_distortion,
      noise     = quality_degradation,
      )

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = dict(frameskip=0.1,  color=8.5,  noise=45.),

      # high-quality encoders
      zlib       = dict(frameskip=0.0,  color=0.0, noise=0.0),
      ffv1       = dict(frameskip=0.05, color=9.,  noise=46.),
      vp8        = dict(frameskip=0.3,  color=9.0, noise=65.),
      libvpx     = dict(frameskip=0.3,  color=9.0, noise=65.),
      h264       = dict(frameskip=0.4,  color=8.5, noise=50.),
      libx264    = dict(frameskip=0.4,  color=8.5, noise=50.),
      theora     = dict(frameskip=0.5,  color=9.0, noise=70.),
      libtheora  = dict(frameskip=0.5,  color=9.0, noise=70.),
      mpeg4      = dict(frameskip=1.0,  color=9.0, noise=55.),

      # older, but still good quality encoders
      mjpeg      = dict(frameskip=1.2,  color=8.5, noise=50.),
      mpegvideo  = dict(frameskip=1.3,  color=8.5, noise=55.),
      mpeg2video = dict(frameskip=1.3,  color=8.5, noise=55.),
      mpeg1video = dict(frameskip=1.4,  color=9.0, noise=50.),

      # low quality encoders - avoid using - available for compatibility
      wmv2       = dict(frameskip=3.0,  color=10., noise=50.),
      wmv1       = dict(frameskip=2.5,  color=10., noise=50.),
      msmpeg4    = dict(frameskip=5.,   color=10., noise=50.),
      msmpeg4v2  = dict(frameskip=5.,   color=10., noise=50.),
      )

  # some exceptions
  if testutils.ffmpeg_version_lessthan('0.6'):
    distortions['ffv1']['frameskip'] = 0.55
    distortions['mpeg1video']['frameskip'] = 1.5
    distortions['mpegvideo']['color'] = 9.0
    distortions['mpegvideo']['frameskip'] = 1.4
    distortions['mpeg2video']['color'] = 9.0
    distortions['mpeg2video']['frameskip'] = 1.4

  from .._externals import supported_videowriter_formats
  SUPPORTED = supported_videowriter_formats()
  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      for method in methods:
        check_format_codec.description = "%s.test_%s_format_%s_codec_%s" % (__name__, method, format, codec)
        distortion = distortions.get(codec, distortions['default'])[method]
        yield check_format_codec, methods[method], shape, framerate, format, codec, distortion

@testutils.ffmpeg_found()
def check_user_video(format, codec, maxdist):

  from .. import VideoReader, VideoWriter
  fname = testutils.temporary_filename(suffix='.%s' % format)
  MAXLENTH = 10 #use only the first 10 frames

  try:

    orig_vreader = VideoReader(INPUT_VIDEO)
    orig = orig_vreader[:MAXLENTH]
    (olength, _, oheight, owidth) = orig.shape
    assert len(orig) == MAXLENTH, "original length %d != %d MAXLENTH" % (len(orig), MAXLENTH)

    # encode the input video using the format and codec provided by the user
    outv = VideoWriter(fname, oheight, owidth, orig_vreader.frame_rate,
        codec=codec, format=format)
    for k in orig: outv.append(k)
    outv.close()

    # reload from saved file
    encoded = VideoReader(fname)
    reloaded = encoded.load()

    # test number of frames is correct
    assert len(orig) == len(encoded), "original length %d != %d encoded" % (len(orig), len(encoded))
    assert len(orig) == len(reloaded), "original length %d != %d reloaded" % (len(orig), len(reloaded))

    # test distortion patterns (quick sequential check)
    dist = []
    for k, of in enumerate(orig):
      dist.append(abs(of.astype('float64')-reloaded[k].astype('float64')).mean())
    assert max(dist) <= maxdist, "max(distortion) %g > %g allowed" % (max(dist), maxdist)

    # make sure that the encoded frame rate is not off by a big amount
    assert abs(orig_vreader.frame_rate - encoded.frame_rate) <= (1.0/MAXLENTH), "original video framerate %g differs from encoded %g by more than %g" % (encoded.frame_rate, framerate, 1.0/MAXLENTH)

  finally:

    if os.path.exists(fname): os.unlink(fname)

def test_user_video():

  # distortion patterns for specific codecs
  distortions = dict(
      # we require high standards by default
      default    = 1.5,

      # high-quality encoders
      zlib       = 0.0,
      ffv1       = 1.7,
      vp8        = 2.7,
      libvpx     = 2.7,
      h264       = 2.5,
      libx264    = 2.5,
      theora     = 2.0,
      libtheora  = 2.0,
      mpeg4      = 2.3,

      # older, but still good quality encoders
      mjpeg      = 1.8,
      mpegvideo  = 2.3,
      mpeg2video = 2.3,
      mpeg1video = 2.3,

      # low quality encoders - avoid using - available for compatibility
      wmv2       = 2.3,
      wmv1       = 2.3,
      msmpeg4    = 2.3,
      msmpeg4v2  = 2.3,
      )

  from .._externals import supported_videowriter_formats
  SUPPORTED = supported_videowriter_formats()
  for format in SUPPORTED:
    for codec in SUPPORTED[format]['supported_codecs']:
      check_user_video.description = "%s.test_user_video_format_%s_codec_%s" % (__name__, format, codec)
      distortion = distortions.get(codec, distortions['default'])
      yield check_user_video, format, codec, distortion
