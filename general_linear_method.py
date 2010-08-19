from ode_solver import ODESolver
import rooted_trees as tt

#=====================================================
class GeneralLinearMethod(ODESolver):
#=====================================================
    """ General class for multstage, multistep methods """
    def __init__(self):
        self.data=None

    def order_conditions(self,p):
        """
        Generate a list of order conditions up to order p 
        """

        forest=tt.recursive_trees(0)
        for i in range(1,p+1): forest+=tt.recursive_trees(i)
        oc={tt.RootedTree(''):''}
        for tree in forest[1:]:
            oc[tree]=self.elementary_weight(tree)-tree.Emap()
        return oc

    def elementary_weight(self,tree):
        print 'Not implemented yet!'
        sys.exit(1)
