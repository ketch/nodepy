"""
Unit tests for NodePy.
This needs to be updated.
"""
from __future__ import absolute_import
import nodepy.linear_multistep_method as lmm
import nodepy.runge_kutta_method as rk
import unittest as ut
import numpy as np

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


#=====================================================
# Bundled methods must have their documented order
#=====================================================

class BundledMethodOrderTest(ut.TestCase):
    """
    Check every bundled method against the order given by its name or
    documentation.  The expected orders below are taken from the method
    names and descriptions, not computed.  Pairs list (order, embedded order);
    the embedded order is checked only where the name or source states it.
    """
    # Floating-point coefficients are stored to about 16 digits, so the
    # residuals of high-order conditions can exceed the default 1e-14.
    tol = 1.e-12

    rk_orders = {
        'BE': 1, 'FE': 1, 'SSP22star': 2,
        'LobattoIIIC4': 6, 'LobattoIIIC3': 4, 'LobattoIIIC2': 2,
        'LobattoIIIA2': 2, 'LobattoIIIA3': 4, 'RadauIIA2': 3, 'RadauIIA3': 5,
        'SDIRK54': 4, 'SDIRK23': 3, 'SDIRK34': 4, 'TR-BDF2': 2,
        'SSP22': 2, 'Heun22': 2, 'SSP33': 3, 'Heun33': 3, 'NSSP32': 2,
        'NSSP33': 3, 'SSP43': 3, 'SSP104': (4, 3),
        'MTE22': 2, 'Mid22': 2, 'RK44': 4, 'BuRK65': 5, 'GL2': 4, 'GL3': 6,
        'Fehlberg45': (5, 4), 'Fehlberg43': (4, 3), 'Merson43': (4, 3),
        u'Söderlind43': (4, 3), 'Soderlind43': (4, 3), 'Zonneveld43': (4, 3),
        'Tsit5': (5, 4), 'CK5': (5, 4), 'HH5': (5, 4), 'HH5S': (5, 4),
        'DP5': (5, 4), 'CMR6': (6, 5), 'BS3': (3, 2), 'BS5': (5, 4),
        'SS3': (3, 2), 'PD8': (8, 7), 'SSP75': 5, 'SSP85': 5, 'SSP95': 5,
        'SSP54': 4, 'SSP53': 3, 'SSP63': 3,
    }
    low_storage_orders = {
        'RK45[2N]': 4, 'DDAS47': 4, 'LDDC46': 4, 'RK34[2R]C': (3, 2),
        'RK45[2R]C': (4, 3), 'RK45[3R]C': (4, 3), 'RK58[3R]C': (5, 4),
        'RK59[2R]S': (5, 4), 'RK35[3S*]': 3, 'RK49[3S*]': 4, 'RK510[3S*]': 5,
    }
    tsrk_orders = {'order3': 3, 'order4': 4, 'order5': 5}
    # ET112 pairs forward Euler (order 1) with the theta method, so its order is 1.
    lmm_orders = {'eBDF5': 5, 'ET112': 1}

    # Methods known not to have their documented order (ketch/nodepy issue 110).
    # Remove an entry when its coefficients are corrected.
    known_problems = set()

    def _check(self, label, method, expected, order):
        import io, contextlib
        expected = expected if isinstance(expected, tuple) else (expected,)
        with contextlib.redirect_stdout(io.StringIO()):  # order() may print warnings
            found = [order(method)]
            if len(expected) == 2:
                found.append(order(method.embedded_method))
        return tuple(found) != expected, (label, expected, tuple(found))

    def test_documented_orders(self):
        from nodepy import lsrk, tsrk
        rk_order = lambda m: m.order(tol=self.tol)
        cases = []
        rks = rk.loadRKM()
        self.assertEqual(set(rks), set(self.rk_orders), 'Update rk_orders for new methods')
        for name, method in rks.items():
            cases.append(self._check(name, method, self.rk_orders[name], rk_order))
        lsrks = lsrk.load_low_storage()
        self.assertEqual(set(lsrks), set(self.low_storage_orders))
        for name, method in lsrks.items():
            cases.append(self._check(name, method, self.low_storage_orders[name], rk_order))
        tsrks = tsrk.loadTSRK()
        self.assertEqual(set(tsrks), set(self.tsrk_orders))
        for name, method in tsrks.items():
            cases.append(self._check('tsrk '+name, method, self.tsrk_orders[name],
                                     lambda m: m.order(tol=self.tol)))

        lmms = lmm.loadLMM()
        self.assertEqual(set(lmms), set(self.lmm_orders))
        for name, method in lmms.items():
            cases.append(self._check('lmm '+name, method, self.lmm_orders[name],
                                     lambda m: m.order()))

        wrong = {label: (expected, found) for bad, (label, expected, found) in cases if bad}
        unexpected = {k: v for k, v in wrong.items() if k not in self.known_problems}
        fixed = self.known_problems - set(wrong)
        self.assertEqual(unexpected, {}, 'Methods without their documented order: '
                         '{label: (expected, found)}')
        self.assertEqual(fixed, set(), 'These now have their documented order; '
                         'remove them from known_problems')


class TSRKStabilityTest(ut.TestCase):
    def test_principal_eigenvalue_approximates_exp(self):
        # For a method of order p, the stability matrix has an eigenvalue
        # within O(z^(p+1)) of exp(z).  This failed when the sign of b in
        # stability_matrix was wrong.
        from nodepy import tsrk
        for name, p in [('order3', 3), ('order4', 4), ('order5', 5)]:
            method = tsrk.loadTSRK(name)
            for z in [-0.1, -0.01]:
                ev = np.linalg.eigvals(method.stability_matrix(z))
                err = np.min(np.abs(ev - np.exp(z)))
                self.assertLess(err, 2*abs(z)**(p+1), (name, z, err))

    def test_order5_real_stability_interval(self):
        # Jackiewicz & Tracogna (1995), p. 1425: the interval of absolute
        # stability is approximately (-0.486, 0).
        from nodepy import tsrk
        method = tsrk.loadTSRK('order5')
        spectral_radius = lambda z: np.max(np.abs(np.linalg.eigvals(method.stability_matrix(z))))
        for z in [-0.48, -0.3, -0.1, -0.01]:
            self.assertLessEqual(spectral_radius(z), 1 + 1.e-12, z)
        self.assertGreater(spectral_radius(-0.49), 1)


#=====================================================
# Tests of the direct evaluation of order conditions
#=====================================================

class RootedTreeTableTest(ut.TestCase):

    def test_number_of_trees(self):
        import nodepy.rooted_trees as rt
        # OEIS A000081
        counts = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766, 12486, 32973]
        self.assertEqual([len(rt.tree_ids(p)) for p in range(1, 15)], counts)

    def test_trees_match_list_trees(self):
        import nodepy.rooted_trees as rt
        def canonical(tree):
            # Trees as nested sorted tuples, so that equal trees compare equal
            nleaves, subtrees = tree._parse_subtrees()
            return tuple(sorted([()]*nleaves + [canonical(s) for s in subtrees]))
        for p in range(1, 9):
            trees = [rt.tree_from_id(i) for i in rt.tree_ids(p)]
            self.assertEqual(sorted(map(canonical, trees)), sorted(map(canonical, rt.list_trees(p))))
            for i, tree in zip(rt.tree_ids(p), trees):
                self.assertEqual(tree.density(), rt.tree_density(i))
                self.assertEqual(tree.symmetry(), rt.tree_symmetry(i))

    def test_lamda_with_extra_arguments(self):
        # lamda must give the same result as lamda_str when the function
        # takes an extra argument.
        import sympy
        import nodepy.rooted_trees as rt
        for p in range(2, 6):
            for tree in rt.list_trees(p):
                trees, factors = tree.lamda(rt.Emap, -1)
                trees_str, factors_str = tree.lamda_str(rt.Emap_str, -1)
                self.assertEqual(trees, trees_str)
                self.assertEqual([sympy.sympify(f) for f in factors],
                                 [sympy.sympify(f) for f in factors_str])


class OrderConditionTest(ut.TestCase):

    def test_residuals_match_exact_arithmetic(self):
        # Floating-point residuals agree with residuals computed exactly
        # from the same (binary) coefficients.
        from fractions import Fraction
        import numpy as np
        exact = lambda x: np.array([Fraction(v) for v in np.ravel(x)], dtype=object).reshape(np.shape(x))
        m = rk.loadRKM('DP5').__num__()
        float_residuals = rk._order_condition_residuals_by_order(m.A, m.b, m.c)
        exact_residuals = rk._order_condition_residuals_by_order(exact(m.A), exact(m.b), exact(m.c))
        for p in range(1, 8):
            zf, ze = next(float_residuals), next(exact_residuals)
            self.assertLess(np.max(np.abs(zf - ze.astype(float))), 1.e-14)

    def test_float_and_exact_order_agree(self):
        for name in ['RK44', 'SSP104', 'GL3', 'RadauIIA3', 'LobattoIIIC4', 'SDIRK34']:
            method = rk.loadRKM(name)
            self.assertEqual(method.order(), method.order(mode='exact'), msg=name)

    def test_high_order_methods(self):
        from nodepy.loadmethod import load_rk_from_file
        for filename, order in [('rk108.txt', 10), ('rk1210.txt', 12), ('rk1412.txt', 14)]:
            method = load_rk_from_file(filename)
            self.assertEqual(method.order(tol=1.e-12), order, msg=filename)
            # The principal error norm needs trees of order p+1 > 10.
            self.assertGreater(method.principal_error_norm(tol=1.e-12), 0.)

    def test_principal_error_norm_matches_error_coefficients(self):
        import numpy as np
        import nodepy.rooted_trees as rt
        dp5 = rk.loadRKM('DP5').__num__()
        coefficients = [dp5.error_coefficient(tree) for tree in rt.list_trees(6)]
        self.assertAlmostEqual(dp5.principal_error_norm(), np.sqrt(np.sum(np.square(coefficients))), 14)

    def test_order_too_large_for_tolerance(self):
        with self.assertRaises(ValueError):
            rk.loadRKM('FE').order(tol=10.)

    def test_effective_order(self):
        self.assertEqual(rk.loadRKM('SSP22').effective_order(), 2)
        self.assertEqual(rk.loadRKM('SSP33').effective_order(), 3)

    def test_twostep_order(self):
        from nodepy import tsrk
        methods = tsrk.loadTSRK()
        self.assertEqual(methods['order3'].order(), 3)
        self.assertEqual(methods['order4'].order(), 4)


class ConstructorTest(ut.TestCase):
    """Subclasses must initialize through their parent classes."""

    def test_low_storage_attributes_from_parent(self):
        from nodepy import lsrk
        for name, method in lsrk.load_low_storage().items():
            self.assertTrue(method.mtype.startswith('Explicit'), msg=name)
            self.assertEqual(len(method.c), len(method), msg=name)
            method._check_consistency()

    def test_low_storage_invalid_arguments(self):
        from nodepy import lsrk
        pair = lsrk.load_low_storage('RK45[2R]C')
        with self.assertRaises(ValueError):
            lsrk.TwoRRungeKuttaMethod(pair.a, pair.b, bhat=pair.bhat)
        betavec, gamma, delta = [0., 1.], [[0., 0.], [0., 1.]], [1., 0.]
        with self.assertRaises(ValueError):
            lsrk.TwoSRungeKuttaMethod(betavec, gamma, delta, '2S_pair')
        with self.assertRaises(ValueError):
            lsrk.TwoSRungeKuttaPair(betavec, gamma, delta, '2S')   # no bhat
        with self.assertRaises(ValueError):
            lsrk.TwoSRungeKuttaMethod(betavec, gamma, delta, '4S')

    def test_continuous_method_keeps_order(self):
        A = np.array([[0., 0.], [1., 0.]])
        b = np.array([0.5, 0.5])
        method = rk.ContinuousExplicitRungeKuttaMethod(A, b, b_dense=np.zeros((2, 3)), order=2)
        self.assertEqual(method._p, 2)


class WarningTest(ut.TestCase):
    """Problems are reported with warnings, not print()."""

    def test_explicit_method_as_RungeKuttaMethod_warns(self):
        A = np.array([[0., 0.], [1., 0.]])
        with self.assertWarns(UserWarning):
            rk.RungeKuttaMethod(A, np.array([0.5, 0.5]))

    def test_apparent_order_zero_warns(self):
        A = np.array([[0., 0.], [1., 0.]])
        with self.assertWarns(UserWarning):
            p = rk.ExplicitRungeKuttaMethod(A, np.array([0.5, 0.6])).order()
        self.assertEqual(p, 0)

    def test_effective_order_limit_warns(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(rk.loadRKM('RK44').effective_order(), 4)


class OptimalSSPCoefficientTest(ut.TestCase):
    """SSP coefficients of the optimal SSP methods, as printed by Ruuth (2006)."""
    known = {'SSP53': 2.65062919143939, 'SSP54': 1.50818004918983,
             'SSP63': 3.51839230899685}

    def test_ssp_coefficients(self):
        for name, value in self.known.items():
            computed = rk.loadRKM(name).absolute_monotonicity_radius(acc=1.e-13)
            self.assertAlmostEqual(computed, value, 11, msg=name)


if __name__== "__main__":
    ut.main()
