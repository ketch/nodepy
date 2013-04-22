from distutils.core import setup

setup(name='nodepy',
      version='0.4',
      package_dir={'nodepy': ''},
      packages=['nodepy'],
      py_modules = ['runge_kutta_method', 'linear_multistep_method', 'rooted_trees',
                    'convergence','general_linear_method','graph','ivp',
                    'low_storage_rk','oc_butcher','oc_butcher_high_order',
                    'ode_solver','semidisc','snp','stability_function',
                    'strmanip','threestep_runge_kutta_method',
                    'twostep_runge_kutta_method','unit_tests','utils']
      #package_data={'' : ['examples/*.py']},
      author=['David Ketcheson'],
      author_email=['dketch@gmail.com'],
      url='http://numerics.kaust.edu.sa/nodepy/',
      description='Numerical ODE solvers in Python',
      license='modified BSD',
      install_requires=['numpy','sympy'],
      )
