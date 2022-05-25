#Make matlab script to evaluate error coefficient
#for RK methods

from nodepy import rooted_trees as rt
from nodepy import runge_kutta_method as rk

p=11

f=open('oc_butcher.m','w')

ioc=1
for ip in range(2,p):
    f.write("if p>="+str(ip))
    f.write("\n  % order "+str(ip)+" conditions:\n")
    forest = rt.list_trees(ip)
    for tree in forest:
        oc=rk.elementary_weight_str_matlab(tree)
        rhs =str(rt.Emap(tree))
        f.write("  coneq("+str(ioc)+") = "+oc[1:-1]+"-"+rhs+";\n")
        ioc+=1
    f.write("end\n\n")

f.close()
