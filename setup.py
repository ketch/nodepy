import setuptools
from distutils.core import setup

# Use README as description on Pypi
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='nodepy',
      version='1.0.1',
      packages=['nodepy'],
      author=['David Ketcheson'],
      author_email='dketch@gmail.com',
      url='https://github.com/ketch/nodepy',
      description='Numerical ODE solvers in Python',
      license='modified BSD',
      install_requires=['numpy','sympy','matplotlib'],
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
