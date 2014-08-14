#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.core']))
from bob.extension.utils import egrep, find_header, find_library
from bob.blitz.extension import Extension, Library, build_ext

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(package_dir, 'bob', 'io', 'base')

version = '2.0.0a0'

def libhdf5_version(header):

  version = egrep(header, r"#\s*define\s+H5_VERSION\s+\"([\d\.]+)\"")
  if not len(version): return None
  return version[0].group(1)

class hdf5:

  def __init__ (self, requirement='', only_static=False):
    """
    Searches for libhdf5 in stock locations. Allows user to override.

    If the user sets the environment variable BOB_PREFIX_PATH, that prefixes
    the standard path locations.

    Parameters:

    requirement, str
      A string, indicating a version requirement for this library. For example,
      ``'>= 8.2'``.

    only_static, boolean
      A flag, that indicates if we intend to link against the static library
      only. This will trigger our library search to disconsider shared
      libraries when searching.
    """

    self.name = 'hdf5'
    header = 'hdf5.h'

    candidates = find_header(header)

    if not candidates:
      raise RuntimeError("could not find %s's `%s' - have you installed %s on this machine?" % (self.name, header, self.name))

    found = False

    if not requirement:
      self.include_directory = os.path.dirname(candidates[0])
      directory = os.path.dirname(candidates[0])
      version_header = os.path.join(directory, 'H5pubconf.h')
      self.version = libhdf5_version(version_header)
      found = True

    else:

      # requirement is 'operator' 'version'
      operator, required = [k.strip() for k in requirement.split(' ', 1)]

      # now check for user requirements
      for candidate in candidates:
        directory = os.path.dirname(candidate)
        version_header = os.path.join(directory, 'H5pubconf.h')
        version = libhdf5_version(version_header)
        available = LooseVersion(version)
        if (operator == '<' and available < required) or \
           (operator == '<=' and available <= required) or \
           (operator == '>' and available > required) or \
           (operator == '>=' and available >= required) or \
           (operator == '==' and available == required):
          self.include_directory = os.path.dirname(candidate)
          self.version = version
          found = True
          break

    if not found:
      raise RuntimeError("could not find the required (%s) version of %s on the file system (looked at: %s)" % (requirement, self.name, ', '.join(candidates)))

    # normalize
    self.include_directory = os.path.normpath(self.include_directory)

    # find library
    prefix = os.path.dirname(os.path.dirname(self.include_directory))
    module = 'hdf5'
    candidates = find_library(module, version=self.version, prefixes=[prefix], only_static=only_static)

    if not candidates:
      raise RuntimeError("cannot find required %s binary module `%s' - make sure libsvm is installed on `%s'" % (self.name, module, prefix))

    # libraries
    self.libraries = []
    name, ext = os.path.splitext(os.path.basename(candidates[0]))
    if ext in ['.so', '.a', '.dylib', '.dll']:
      self.libraries.append(name[3:]) #strip 'lib' from the name
    else: #link against the whole thing
      self.libraries.append(':' + os.path.basename(candidates[0]))

    # library path
    self.library_directory = os.path.dirname(candidates[0])

  def macros(self):
    return [
        ('HAVE_%s' % self.name.upper(), '1'),
        ('%s_VERSION' % self.name.upper(), '"%s"' % self.version),
        ]


hdf5_pkg = hdf5()

extra_compile_args = [
    '-isystem', hdf5_pkg.include_directory,
    ]

library_dirs = [
    hdf5_pkg.library_directory,
    ]

libraries = hdf5_pkg.libraries

define_macros = hdf5_pkg.macros()


setup(

    name='bob.io.base',
    version=version,
    description='Base bindings for bob.io',
    url='http://github.com/bioidiap/bob.io.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=[
      'setuptools',
      'bob.blitz',
    ],

    namespace_packages=[
      "bob",
      "bob.io",
      ],

    ext_modules = [
      Extension("bob.io.base.version",
        [
          "bob/io/base/version.cpp",
          ],
        define_macros = define_macros,
        extra_compile_args = extra_compile_args,
        version = version,
        bob_packages = ['bob.core'],
        packages = ['boost'],
        boost_modules = ['system'],
        ),

      Library("bob_io_base",
        [
          "bob/io/base/cpp/CodecRegistry.cpp",
          "bob/io/base/cpp/CSVFile.cpp",
          "bob/io/base/cpp/File.cpp",
          "bob/io/base/cpp/HDF5ArrayFile.cpp",
          "bob/io/base/cpp/HDF5Attribute.cpp",
          "bob/io/base/cpp/HDF5Dataset.cpp",
          "bob/io/base/cpp/HDF5File.cpp",
          "bob/io/base/cpp/HDF5Group.cpp",
          "bob/io/base/cpp/HDF5Types.cpp",
          "bob/io/base/cpp/HDF5Utils.cpp",
          "bob/io/base/cpp/reorder.cpp",
          "bob/io/base/cpp/T3File.cpp",
          "bob/io/base/cpp/TensorArrayFile.cpp",
          "bob/io/base/cpp/TensorFileHeader.cpp",
          "bob/io/base/cpp/utils.cpp",
          "bob/io/base/cpp/TensorFile.cpp",
          "bob/io/base/cpp/array.cpp",
          "bob/io/base/cpp/array_type.cpp",
          "bob/io/base/cpp/blitz_array.cpp",
        ],
        package_directory = package_dir,
        target_directory = target_dir,
        libraries = libraries,
        library_dirs = library_dirs,
        include_dirs = [hdf5_pkg.include_directory],
        define_macros = define_macros,
        version = version,
        bob_packages = ['bob.core', 'bob.blitz'],
        packages = ['boost'],
        boost_modules = ['system'],
      ),

      Extension("bob.io.base._library",
        [
          "bob/io/base/bobskin.cpp",
          "bob/io/base/codec.cpp",
          "bob/io/base/file.cpp",
          "bob/io/base/hdf5.cpp",
          "bob/io/base/main.cpp",
          ],
        library_dirs = library_dirs,
        libraries = libraries + ['bob_io_base'],
        define_macros = define_macros,
        extra_compile_args = extra_compile_args,
        version = version,
        bob_packages = ['bob.core'],
        packages = ['boost'],
        boost_modules = ['system'],
        ),
      ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
