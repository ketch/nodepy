#Make python script to evaluate order conditions
#for RK methods

from nodepy import rooted_trees as rt
from nodepy import runge_kutta_method as rk

p_max=15

f=open('oc_butcher.py','w')

f.write("def order(rk,tol=1.e-13):\n")
f.write("    from numpy import dot,zeros,all,sum,abs\n")
f.write("    coneq = zeros((1000))\n")
f.write("    A=rk.A\n    b=rk.b\n    c=rk.c\n")
f.write("    coneq[0]=sum(b)-1.\n")
f.write("    if any(abs(coneq)>tol):\n")
f.write("        return 0")
for ip in range(2,p_max):
    print 'Generating order '+str(ip)+' conditions...'
    ioc=0
    f.write("\n    # order "+str(ip)+" conditions:\n")
    forest = rt.list_trees(ip)
    for tree in forest:
        oc=rk.elementary_weight_str(tree,style='python')
        rhs =str(rt.Emap(tree))
        f.write("    coneq["+str(ioc)+"]="+oc+"-"+rhs+".\n")
        ioc+=1
    f.write("    if any(abs(coneq)>tol):\n")
    f.write("        return "+str(ip-1))

f.close()
