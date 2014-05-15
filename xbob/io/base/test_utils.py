#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu Feb  7 09:58:22 2013
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Re-usable decorators and utilities for xbob test code
"""

import os
import functools
import nose.plugins.skip
from distutils.version import StrictVersion as SV

def datafile(f, module=None, path='data'):
  """Returns the test file on the "data" subdirectory of the current module.

  Keyword attributes

  f: str
    This is the filename of the file you want to retrieve. Something like
    ``'movie.avi'``.

  package: string, optional
    This is the python-style package name of the module you want to retrieve
    the data from. This should be something like ``xbob.io.test``, but you
    normally refer it using the ``__name__`` property of the module you want to
    find the path relative to.

  path: str, optional
    This is the subdirectory where the datafile will be taken from inside the
    module. Normally (the default) ``data``. It can be set to ``None`` if it
    should be taken from the module path root (where the ``__init__.py`` file
    sits).

  Returns the full path of the file.
  """

  resource = __name__ if module is None else module
  final_path = f if path is None else os.path.join(path, f)
  return __import__('pkg_resources').resource_filename(resource, final_path)

def temporary_filename(prefix='bobtest_', suffix='.hdf5'):
  """Generates a temporary filename to be used in tests"""

  (fd, name) = __import__('tempfile').mkstemp(suffix, prefix)
  os.close(fd)
  os.unlink(name)
  return name

def bob_at_least(version_geq):
  '''Decorator to check if at least a certain version of Bob is installed

  To use this, decorate your test routine with something like:

  .. code-block:: python

    @bob_at_least('1.2.2')

  '''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      from .version import externals
      inst = SV(externals['Bob'][0])
      if inst < version_geq:
        raise nose.plugins.skip.SkipTest('Bob version installed (%s) is smaller than required for this test (%s)' % (externals['Bob'][0], version_geq))
      return test(*args, **kwargs)

    return wrapper

  return test_wrapper

def extension_available(extension):
  '''Decorator to check if a extension is available before enabling a test'''

  def test_wrapper(test):

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
      from .version import extensions
      if extension in extensions():
        return test(*args, **kwargs)
      else:
        raise nose.plugins.skip.SkipTest('Extension to handle "%s" files was not available at compile time' % extension)

    return wrapper

  return test_wrapper
