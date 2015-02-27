from distutils.core import setup

setup(name='nodepy',
      version='0.6',
      packages=['nodepy'],
      author=['David Ketcheson'],
      author_email=['dketch@gmail.com'],
      url='http://numerics.kaust.edu.sa/nodepy/',
      description='Numerical ODE solvers in Python',
      license='modified BSD',
      install_requires=['numpy','sympy'],
      )
