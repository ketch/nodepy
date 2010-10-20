import nodepy.runge_kutta_method as rk
import nodepy.convergence as cv
from nodepy import ivp

#Load some methods:
rk4=rk.loadRKM('RK44')
SSP2=rk.loadRKM('SSP22')
SSP104=rk.loadRKM('SSP104')

#Define an IVP:
#myivp=ivp.exp_fun(1.)
myivp=ivp.load_ivp('test')

#Start and end time:
T=[0.,5.]

cv.ctest([rk4,SSP2,SSP104],myivp,T)
