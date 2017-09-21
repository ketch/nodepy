from __future__ import absolute_import
from nodepy.ode_solver import ODESolver
import nodepy.rooted_trees as rt
from six.moves import range

#=====================================================
class GeneralLinearMethod(ODESolver):
#=====================================================
    """ General class for multstage, multistep methods.

        This is a pure virtual class; only its child classes
        should be instantiated.
    """
    def __init__(self):
        self.data=None

    def order_conditions(self,p):
        """
        Generate a list of order conditions up to order p
        """

        forest=rt.list_trees(0)
        for i in range(1,p+1): forest+=rt.list_trees(i)
        oc={rt.RootedTree(''):''}
        for tree in forest[1:]:
            oc[tree]=self.elementary_weight(tree)-rt.Emap(tree)
        return oc

    def elementary_weight(self,tree):
        raise NotImplementedError
