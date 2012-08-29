from distutils.core import setup

setup(name='nodepy',
      version='0.4',
      package_dir={'nodepy': ''},
      packages=['nodepy'],
      #package_data={'' : ['examples/*.py']},
      author=['David Ketcheson'],
      author_email=['dketch@gmail.com'],
      url='http://numerics.kaust.edu.sa/nodepy/',
      description='Numerical ODE solvers in Python',
      license='modified BSD',
      requires=['numpy','sympy'],
      )
