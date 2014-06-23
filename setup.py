#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz']))
from bob.blitz.extension import Extension

import os
package_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.join(package_dir, 'bob', 'io', 'base', 'include')
include_dirs = [package_dir]

packages = ['bob-io >= 2.0.0a2']
version = '2.0.0a0'

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
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      Extension("bob.io.base._library",
        [
          "bob/io/base/bobskin.cpp",
          "bob/io/base/codec.cpp",
          "bob/io/base/file.cpp",
          "bob/io/base/hdf5.cpp",
          "bob/io/base/main.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      ],

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
