#!/usr/bin/env python

"""Setup script for the sample #1 module distribution:
   single top-level pure Python module, named explicitly
   in 'py_modules'.


https://www.ibm.com/developerworks/linux/library/l-cpmod/index.html

python3 setup.py sdist
   """

from distutils.core import setup

setup (# Distribution meta-data
       name = "tmm",
       version = "1.0",
       description = "Distutils sample distribution #1",

# Description of modules and packages in the distribution
       py_modules = ['preprocessing'],
      )
