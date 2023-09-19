.. _installation:

Installation
------------

There are multiple methods for installing oakutils. The recommended method is
to install oakutils into a virtual environment. This will ensure that the
oakutils dependencies are isolated from other Python projects you may be
working on.

Methods:
--------
#. Pip:

   .. code-block:: console

      $ pip3 install oakutils

#. From source:

   .. code-block:: console

      $ git clone https://github.com/justincdavis/oakutils.git
      $ cd oakutils
      $ pip3 install .

Optional Dependencies:
----------------------
#. compiler:

   .. code-block:: console

      $ pip3 install oakutils[compiler]
   
   This will install dependencies allowing the use of the compiler module.

#. dev:

   .. code-block:: console

      $ pip3 install oakutils[dev]
   
   This will install dependencies allowing a full development environment.
   All CI and tools used for development will be installed and can be run.
