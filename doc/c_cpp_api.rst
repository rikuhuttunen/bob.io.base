.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Tue 15 Oct 14:59:05 2013

=========
 C++ API
=========

The C++ API of ``xbob.io`` allows users to leverage from automatic converters
for classes in :py:class:`xbob.io`.  To use the C API, clients should first,
include the header file ``<xbob.io/api.h>`` on their compilation units and
then, make sure to call once ``import_xbob_io()`` at their module
instantiation, as explained at the `Python manual
<http://docs.python.org/2/extending/extending.html#using-capsules>`_.

Here is a dummy C example showing how to include the header and where to call
the import function:

.. code-block:: c++

   #include <xbob.io/api.h>

   PyMODINIT_FUNC initclient(void) {

     PyObject* m Py_InitModule("client", ClientMethods);

     if (!m) return;

     // imports the NumPy C-API 
     import_array();

     // imports blitz.array C-API
     import_blitz_array();

     // imports xbob.core.random C-API
     import_xbob_io();

   }

.. note::

  The include directory can be discovered using
  :py:func:`xbob.io.get_include`.

Generic Functions
-----------------

.. cpp:function:: int PyBobIo_AsTypenum(bob::core::array::ElementType et)

   Converts the input Bob element type into a ``NPY_<TYPE>`` enumeration value.
   Returns ``NPY_NOTYPE`` in case of problems, and sets a
   :py:class:`RuntimeError`.

Bob File Support
----------------

.. cpp:type:: PyBobIoFileObject

   The pythonic object representation for a ``bob::io::File`` object.

   .. code-block:: cpp

      typedef struct {
        PyObject_HEAD
        boost::shared_ptr<bob::io::File> f;
      } PyBobIoFileObject;

   .. cpp:member:: boost::shared_ptr<bob::io::File> f

      A pointer to a file being read or written.

.. include:: links.rst
