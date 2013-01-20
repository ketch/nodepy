#Make matlab script to evaluate error coefficient
#for RK methods

from nodepy import rooted_trees as rt
from nodepy import runge_kutta_method as rk

p=10

f=open('errcoeff.m','w')

f.write("function D=errcoeff(x,class,s,p)\n\n")
f.write("[A,b,c]=unpack_rk(x,s,class);\n\n")

for ip in range(1,p):
    ioc=1
    f.write("elseif p=="+str(ip))
    f.write("\n  % order "+str(ip+1)+" conditions:\n")
    forest = rt.list_trees(ip+1)
    for tree in forest:
        oc=rk.elementary_weight_str(tree,style='matlab')
        rhs =str(rt.Emap(tree))
        #f.write("  tau("+str(ioc)+")="+oc+"-"+rhs+";\n")
        f.write("  tau("+str(ioc)+")=("+oc+"-"+rhs+")/"+str(tree.symmetry())+";\n")
        ioc+=1

f.close()
