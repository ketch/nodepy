from ode_solver import ODESolver
import rooted_trees as rt

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

        forest=rt.recursive_trees(0)
        for i in range(1,p+1): forest+=rt.recursive_trees(i)
        oc={rt.RootedTree(''):''}
        for tree in forest[1:]:
            oc[tree]=self.elementary_weight(tree)-rt.Emap(tree)
        return oc

    def elementary_weight(self,tree):
        raise NotImplementedError
