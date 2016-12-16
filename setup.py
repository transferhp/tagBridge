#!/usr/bin/env python
# Email: peng.hao@student.uts.edu.au
# Author: Peng Hao
# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize

setup(
		ext_modules = cythonize("./model/svd.pyx")
	)
