#Make matlab script to evaluate order conditions
#for 3-step RK methods

import rooted_trees as tt
import threestep_runge_kutta_method as thsrk

p=6

f=open('oc_thsrk.m','w')

f.write("function coneq=oc_thsrk(x,class,s,p)\n\n")
f.write("e=ones(s,1);\n")
f.write("n=set_n_thsrk(class,s);\n\n")
f.write("% Code to extract coefficient arrays from x here\n\n")
f.write("% Order conditions\n")

ioc=1
for ip in range(1,p+1):
    f.write("\n% order "+str(ip)+" conditions:\n")
    code=thsrk.ThSRKOrderConditions(ip)
    for oc in code:
        f.write("coneq("+str(ioc)+")="+oc+";\n")
        ioc+=1

f.write("\nend")
f.close()
