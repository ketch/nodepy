"""
Unit tests for NodePy.
This needs to be updated.
"""
import linear_multistep_method as lmm
import runge_kutta_method as rk
import unittest as ut

class LinearMultistepTestCase(ut.TestCase):
    pass

class LMMOrderTest(LinearMultistepTestCase):
    def runTest(self):
        for k in range(1,7):
            ab=lmm.Adams_Bashforth(k)
            self.assertEqual(ab.order(),k)
            am=lmm.Adams_Moulton(k)
            self.assertEqual(am.order(),k+1)
            bdf=lmm.backward_difference_formula(k)
            self.assertEqual(bdf.order(),k)

class LMMSSPCoeffTest(LinearMultistepTestCase):
    def runTest(self):
        for k in range(2,100):
            ssp2=lmm.elmm_ssp2(k)
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
            self.assertAlmostEqual(self.RKs[method].absolute_monotonicity_radius(),SSPCoefficient,9)

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

if __name__== "__main__":
    ut.main()
