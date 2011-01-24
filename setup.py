from distutils.core import setup

setup(name='nodepy',
      version='0.3',
      package_dir={'nodepy': ''},
      packages=['nodepy'],
      package_data={'' : ['examples/*.py']},
      author=['David Ketcheson'],
      author_email=['dketch@gmail.com'],
      url='http://web.kaust.edu.sa/faculty/davidketcheson/NodePy/',
      description='Numerical ODE solvers in Python',
      license='BSD',
      requires=['numpy'],
      )
