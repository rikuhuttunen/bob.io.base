#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['boost', 'hdf5']
boost_modules = ['system', 'filesystem']

setup(

    name='bob.io.base',
    version=version,
    description='Basic IO for Bob',
    url='http://github.com/bioidiap/bob.io.base',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,

    namespace_packages=[
      "bob",
      "bob.io",
    ],

    ext_modules = [
      Extension("bob.io.base.version",
        [
          "bob/io/base/version.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Library("bob.io.base.bob_io_base",
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
        version = version,
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
      ),

      Extension("bob.io.base._library",
        [
          "bob/io/base/bobskin.cpp",
          "bob/io/base/codec.cpp",
          "bob/io/base/file.cpp",
          "bob/io/base/hdf5.cpp",
          "bob/io/base/main.cpp",
        ],
        version = version,
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
