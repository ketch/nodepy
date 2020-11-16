import setuptools
from distutils.core import setup

setup(name='nodepy',
      version='1.0.1',
      packages=['nodepy'],
      author=['David Ketcheson'],
      author_email='dketch@gmail.com',
      url='https://github.com/ketch/nodepy',
      description='Numerical ODE solvers in Python',
      license='modified BSD',
      install_requires=['numpy','sympy','matplotlib'],
      )
