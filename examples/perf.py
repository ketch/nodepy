"""Runs a performance test over the non-stiff DETEST suite of problems"""
from nodepy import *
import matplotlib.pyplot as pl

bs5=rk.loadRKM('BS5')
f5=rk.loadRKM('Fehlberg45')
dp5=rk.loadRKM('DP5')
ivps=ivp.detest_suite()

tols=list(map(lambda x:10**-x,range(4,10)))

conv.ptest([bs5,dp5,f5],ivps,tols)
pl.show()
