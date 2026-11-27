"""
Unit tests for NodePy.
This needs to be updated.
"""
from __future__ import absolute_import
import nodepy.linear_multistep_method as lmm
import nodepy.runge_kutta_method as rk
import unittest as ut

class LinearMultistepTestCase(ut.TestCase):
    pass

class LMMOrderTest(LinearMultistepTestCase):
    def runTest(self):
        for k in range(1,7):
            ab = lmm.Adams_Bashforth(k)
            self.assertEqual(ab.order(),k)
            am = lmm.Adams_Moulton(k)
            self.assertEqual(am.order(),k+1)
            bdf = lmm.backward_difference_formula(k)
            self.assertEqual(bdf.order(),k)

class LMMSSPCoeffTest(LinearMultistepTestCase):
    def runTest(self):
        for k in range(2,100):
            ssp2 = lmm.elm_ssp2(k)
            self.assertAlmostEqual(ssp2.ssp_coefficient(),(k-2.)/(k-1.),10)

class RungeKuttaTestCase(ut.TestCase):
    def setUp(self):
        self.RKs=rk.loadRKM()

class RKOrderTest(RungeKuttaTestCase):
    knownValues = ( ('FE',1),
            ('SSP22',2),
            ('SSP33',3),
            ('Mid22',2),
            ('RK44',4),
            ('SSP104',4),
            ('GL2',4),
            ('GL3',6),
            ('BuRK65',5) )

    def runTest(self):
        for method, order in self.knownValues:
            self.assertEqual(self.RKs[method].order(),order)

class RKStageOrderTest(RungeKuttaTestCase):
    knownValues = ( ('FE',1),
            ('SSP22',1),
            ('SSP33',1),
            ('Mid22',1),
            ('RK44',1),
            ('SSP104',1),
            ('GL2',2),
            ('GL3',3),
            ('BuRK65',1) )

    def runTest(self):
        for method, stageorder in self.knownValues:
            self.assertEqual(self.RKs[method].stage_order(),stageorder)

class RKAmradTest(RungeKuttaTestCase):
    knownValues = ( ('FE',1),
            ('SSP22',1),
            ('SSP33',1),
            ('Mid22',0),
            ('RK44',0) ,
            ('SSP104',6),
            ('BuRK65',0) )

    def runTest(self):
        for method, SSPCoefficient in self.knownValues:
            self.assertAlmostEqual(self.RKs[method].absolute_monotonicity_radius(),SSPCoefficient,5)

class linAmradTest(RungeKuttaTestCase):
    knownValues = ( ('FE',1),
            ('SSP22',1),
            ('SSP33',1),
            ('Mid22',1),
            ('RK44',1) ,
            ('SSP104',6),
            ('BuRK65',16/9.) )

    def runTest(self):
        for method, R in self.knownValues:
            self.assertAlmostEqual(self.RKs[method].linear_absolute_monotonicity_radius(),R,2)

class RKstabfuntest(RungeKuttaTestCase):
    import sympy
    one = sympy.Rational(1)
    knownValues = ( ('RK44',[one/24,one/6,one/2,one,one]), )
    formula_representation = (('det',True),('lts',False),('pow',False))

    def runTest(self):
        for method, polycoeffs in self.knownValues:
            for formula, use_butcher in self.formula_representation:
                p,q = self.RKs[method].stability_function(formula=formula,use_butcher=use_butcher)
                assert(all(p.coeffs==polycoeffs))
        ssp2 = rk.SSPRK2(3)
        one = self.one
        for formula in ['lts','pow']:
                p,q = ssp2.stability_function(formula=formula,use_butcher=False)
                assert(all(p.coeffs==[one/12,one/2,one,one]))


#=====================================================
# Regression tests for time stepping
#=====================================================

def _decay_problem(T=1.0, u0=1.0):
    """u' = -u, whose solution is u0*exp(-t)."""
    import numpy as np
    from nodepy.ivp import IVP
    return IVP(f=lambda t, u: -u, u0=np.array([u0]), T=T)


class ODESolverTest(ut.TestCase):

    def test_adaptive_with_output_times(self):
        # Accepted steps between output times must advance the solution.
        import numpy as np
        dp5 = rk.loadRKM('DP5')
        t, u = dp5(_decay_problem(T=[0.5, 1.0]), errtol=1.e-8, dt=0.01)
        self.assertEqual(list(t), [0, 0.5, 1.0])
        for ti, ui in zip(t, u):
            self.assertAlmostEqual(np.atleast_1d(ui)[0], np.exp(-ti), 7)

    def test_initial_time_from_ivp(self):
        # The solver must start from ivp.t0 unless told otherwise.
        from nodepy.ivp import load_ivp
        problem = load_ivp('ode2')
        t, u = rk.loadRKM('RK44')(problem, N=100)
        self.assertEqual(t[0], problem.t0)
        self.assertAlmostEqual(u[-1][0]/problem.exact(problem.T), 1., 4)

    def test_fixed_step_count(self):
        # Rounding errors in t must not produce an extra tiny step.
        t, u = rk.loadRKM('RK44')(_decay_problem(), N=10)
        self.assertEqual(len(t), 11)
        self.assertEqual(t[-1], 1.0)

    def test_dense_output_method_with_scalar_T(self):
        import numpy as np
        ssp2 = rk.SSPRK2(4)
        t, u = ssp2(_decay_problem(), N=20)
        self.assertEqual(len(t), 21)
        self.assertAlmostEqual(u[-1][0], np.exp(-1.), 3)

    def test_max_steps_raises(self):
        with self.assertRaises(RuntimeError):
            rk.loadRKM('DP5')(_decay_problem(T=10.), errtol=1.e-12, dt=1.e-3, max_steps=10)

    def test_invalid_options_raise(self):
        problem = _decay_problem()
        with self.assertRaises(ValueError):
            rk.loadRKM('DP5')(problem, errtol=1.e-6, controllertype='XYZ')
        with self.assertRaises(ValueError):  # No error estimator
            rk.loadRKM('RK44')(problem, errtol=1.e-6)
        with self.assertRaises(NotImplementedError):  # Implicit method
            rk.loadRKM('GL2')(problem)

    def test_one_rhs_evaluation_per_stage(self):
        import numpy as np
        for name in ['RK44', 'DP5', 'SSP104']:
            method = rk.loadRKM(name).__num__()
            for use_butcher in [False, True]:
                count = [0]
                def f(t, u):
                    count[0] += 1
                    return -u
                method.__step__(f, 0., np.array([1.]), 0.1, use_butcher=use_butcher)
                self.assertEqual(count[0], len(method))

    def test_extra_rhs_argument(self):
        import numpy as np
        from nodepy.ivp import IVP
        problem = IVP(f=lambda t, u, lam: lam*u, u0=np.array([1.]), T=1.0)
        for kwargs in [dict(N=20), dict(errtol=1.e-8, dt=0.01)]:
            t, u = rk.loadRKM('DP5')(problem, x=-1.0, **kwargs)
            self.assertAlmostEqual(u[-1][0], np.exp(-1.), 7)


class LowStorageTest(ut.TestCase):

    def _methods(self):
        import nodepy
        from nodepy import lsrk
        path = nodepy.__path__[0]+'/method_coefficients/'
        methods = dict(lsrk.load_low_storage())
        methods['58-2S_acc'] = lsrk.load_LSRK(path+'58-2S_acc.txt', has_emb=True)
        methods['58-3Sstar_acc'] = lsrk.load_LSRK(path+'58-3Sstar_acc.txt', lstype='3S*')
        return methods

    def test_low_storage_step_matches_butcher_form(self):
        import numpy as np
        f = lambda t, u: np.array([u[1], -u[0] + np.cos(t)])
        u0 = np.array([1., 0.5])
        for name, method in self._methods().items():
            m = method.__num__()
            if isinstance(m, rk.ExplicitRungeKuttaPair):
                u1, err1 = m.__step__(f, 0.3, u0, 0.1, estimate_error=True)
                u2, err2 = rk.ExplicitRungeKuttaPair.__step__(m, f, 0.3, u0, 0.1,
                                    estimate_error=True, use_butcher=True)
                self.assertAlmostEqual(err1, err2, 12, msg=name)
            else:
                u1 = m.__step__(f, 0.3, u0, 0.1)
                u2 = rk.ExplicitRungeKuttaMethod.__step__(m, f, 0.3, u0, 0.1, use_butcher=True)
            self.assertLess(np.max(np.abs(u1-u2)), 1.e-12, msg=name)

    def test_low_storage_solve(self):
        import numpy as np
        for name, method in self._methods().items():
            t, u = method(_decay_problem(), N=20)
            self.assertAlmostEqual(u[-1][0], np.exp(-1.), 5, msg=name)
            if hasattr(method, 'embedded_method'):
                t, u = method(_decay_problem(), errtol=1.e-8, dt=0.01)
                self.assertAlmostEqual(u[-1][0], np.exp(-1.), 6, msg=name)


#=====================================================
# Regression tests for method properties
#=====================================================

class RKPropertiesTest(ut.TestCase):

    def test_error_coefficient(self):
        import sympy
        import nodepy.rooted_trees as rt
        rk4 = rk.loadRKM('RK44')
        self.assertEqual(rk4.error_coefficient(rt.list_trees(5)[0]), -sympy.Rational(1, 720))
        self.assertAlmostEqual(rk4.principal_error_norm(), float(sympy.sqrt(1745)/2880), 12)

    def test_dj_reduce_keeps_stages_used_by_embedded_method(self):
        import copy
        import numpy as np
        for name in ['BS3', 'DP5']:
            pair = rk.loadRKM(name)
            self.assertEqual(pair._dj_reducible_stages(), [])
            self.assertEqual(len(copy.deepcopy(pair).dj_reduce()), len(pair))
        # A stage that neither solution uses is removed:
        A = np.array([[0,0,0],[1,0,0],[0.5,0.5,0]])
        pair = rk.ExplicitRungeKuttaPair(A=A, b=np.array([0.5,0.5,0]), bhat=np.array([1.,0,0]))
        reduced = pair.dj_reduce()
        self.assertEqual(len(reduced), 2)
        self.assertEqual(list(reduced.bhat), [1., 0.])

    def test_compose_exact(self):
        import sympy
        fe = rk.loadRKM('FE')
        fe2 = fe*fe
        self.assertEqual(list(fe2.b), [sympy.Rational(1,2), sympy.Rational(1,2)])
        self.assertEqual(fe2.A[1,0], sympy.Rational(1,2))

    def test_discrete_adjoint_requires_nonzero_weights(self):
        with self.assertRaises(ValueError):
            rk.discrete_adjoint(rk.loadRKM('Heun33'))

    def test_print_exact_polynomials(self):
        p, q = rk.loadRKM('RK44').stability_function()
        self.assertIn('0.04167 x', str(p))
        rho, sigma = lmm.Adams_Bashforth(2).characteristic_polynomials()
        self.assertIn('1.5 x', str(sigma))

    def test_imaginary_stability_interval_of_A_stable_method(self):
        import numpy as np
        from nodepy import stability_function
        p, q = rk.loadRKM('BE').stability_function(mode='float')
        self.assertEqual(stability_function.imaginary_stability_interval(p, q), np.inf)


class LMMPropertiesTest(ut.TestCase):

    def test_ssp_coefficient_of_backward_euler(self):
        import sympy
        self.assertEqual(lmm.backward_difference_formula(1).ssp_coefficient(), sympy.oo)


class IVPTest(ut.TestCase):

    def test_vdp_has_no_exact_solution(self):
        from nodepy.ivp import load_ivp
        self.assertIsNone(load_ivp('vdp').exact)


if __name__== "__main__":
    ut.main()
