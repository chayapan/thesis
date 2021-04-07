#!/usr/bin/env python

"""Setup script for the sample #1 module distribution:
   single top-level pure Python module, named explicitly
   in 'py_modules'.

https://docs.python.org/3/distutils/introduction.html

Read the instruction at
https://python.readthedocs.io/en/stable/distutils/setupscript.html

python3 setup.py sdist


python setup.py bdist_wininst for Windows

   """

from distutils.core import setup

setup (# Distribution meta-data
       name = "tmm",
       version = "1.0",
       description = "Distutils sample distribution #1",
       author='Chayapan Khannabha',
       author_email='chayapan@gmail.com',

# Description of modules and packages in the distribution
       # package_dir = {'': 'tmm1'},  # Place everything in dist-packages/tmm1
       py_modules = ['preprocessing',
                     'data.normalize'],
      )
