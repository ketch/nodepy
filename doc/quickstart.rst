==============================
Quick Start Guide
==============================

.. .. contents::

Obtaining NodePy
================

It is possible to install NodePy via pip, but the pip version is
often outdated and the development version is recommended instead.
The current development version of NodePy can be obtained via Git::
    
    git clone git://github.com/ketch/nodepy.git


Installing NodePy
====================

After downloading, simply add the directory
containing the nodepy directory to your Python path.  For instance, if
your nodepy directory is */user/home/python/nodepy*, the appropriate
bash command is::

    $ export PYTHONPATH=/user/home/python/

You will probably want to add this command to your :file:`.bash_profile` file to
avoid retyping it.

NodePy Documentation
====================

NodePy documentation can be found at 
http://numerics.kaust.edu.sa/nodepy

The documentation is also included in the nodepy/doc directory, and can
be built from your local install, if you have Sphinx.

Examples
====================

NodePy comes with some canned examples that can be run to confirm
your installation and to demonstrate capabilities of NodePy.
These can be found in the directory :file:`nodepy/examples`.  
Additional examples can be found in the `User Guide <userguide.html>`_ .
