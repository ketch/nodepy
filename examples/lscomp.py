import NodePy.runge_kutta_method as rk
import NodePy.low_storage_rk as lsrk
import NodePy.convergence as cv
from NodePy.ode import *

rk44=rk.loadRKM('RK44')
#LDDC46=lsrk.load_2R('LDDC46')
#ddas47=lsrk.load_2R('DDAS47')
path = '/Users/ketch/Research/Projects/lowstorageRK/code'
RK452R=lsrk.load_2R('RK45[2R]C')
ls44 =lsrk.load_LSRK(path+'/RK2S-44')
ls75 =lsrk.load_LSRK(path+'/RK2S-75')
ls54s=lsrk.load_LSRK(path+'/RK2Sstar-54',type='2S*')
ls105s=lsrk.load_LSRK('/Users/ketch/research/ODE/lowstorageRK/code/RK2Sstar-105',type='2S*')
ls573s=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK5()7[3S*]A',type='3S*')
ls573sb=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK5()7[3S*]S',type='3S*')
ls613s=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK6()13[3S*]',type='3S*')
myCalvo46=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/myCalvo46[3S*]',type='3S*')
rk453s=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK4()5[3S*]',type='3S*')
rk443s=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK4()4[3S*]',type='3S*')
rk443sb=lsrk.load_LSRK(r'/Users/ketch/research/ODE/lowstorageRK/code/RK4()4[3S*]b',type='3S*')
#methods=[LDDC46,ddas47,ls44,ls75,ls54s,ls105s,ls573s,ls573sb,ls613s,myCalvo46]
methods=[LDDC46,ddas47,ls44,RK452R,myCalvo46,rk453s,rk443s,rk44]

myode=nlsin_fun()
T=[0.,5.]
u0=1.
cv.ctest(methods,myode,T)

for method in methods:
    print method.name, method.principal_error_norm()
