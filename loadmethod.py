import numpy as np
from nodepy import rk

def load_rkpair_from_file(filename='rk1210.txt'):

    f = open(filename)
    #if ((filename=='rk108.txt')):
	#stages=17
    if ((filename=='rk108.txt')):
	name='Feagin\'s 10(8) pair'
	stages=20
    if ((filename=='rk108curtis.txt')):
	name='Curtis\' 10(8) pair'
	stages=21
    if ((filename=='rk1210.txt')):
        name='Feagin\'s 12(10) pair'
	stages=25
    if ((filename=='rk129hiroshi.txt')):
        name='Hiroshi\'s 12(9) pair'
	stages=29
    if ((filename=='rk1412.txt')):
	name='Feagin\'s 14(12) pair'
	stages=35
    b = np.zeros((stages))
    bhat = np.zeros((stages))
    A = np.zeros((stages,stages))
    c = np.zeros((stages))

    for line in f.readlines():
        words = line.split()
        if words == []:
            continue

        if words[-1]=='b[k]':
            current_array = 'b'
        elif words[-1]=='c[k]':
            current_array = 'c'
	elif words[-1]=='bhat[k]':
            current_array = 'bhat'
        elif words[-1]=='A[k,j]':
            current_array = 'A'

        try:
            k = int(words[0])
            if current_array == 'b':
                b[k] = float(words[1])
	    elif current_array =='bhat':
                bhat[k] = float(words[1])
            elif current_array == 'A':
                j = int(words[1])
                A[k,j] = float(words[2])
        except ValueError:
            continue

    return rk.ExplicitRungeKuttaPair(A=A, b=b, bhat=bhat, name=name)

def load_rkmethod_from_file(filename='rk1210.txt'):

    f = open(filename)
    #if ((filename=='rk108.txt')):
	#stages=17
    if ((filename=='rk108.txt')):
	name='Feagin\'s 10(8) pair'
	stages=20
    if ((filename=='rk108curtis.txt')):
	name='Curtis\' 10(8) pair'
	stages=21
    if ((filename=='rk1210.txt')):
        name='Feagin\'s 12(10) pair'
	stages=25
    if ((filename=='rk129hiroshi.txt')):
        name='Hiroshi\'s 12(9) pair'
	stages=29
    if ((filename=='rk1412.txt')):
	name='Feagin\'s 14(12) pair'
	stages=35
    b = np.zeros((stages))
    bhat = np.zeros((stages))
    A = np.zeros((stages,stages))
    c = np.zeros((stages))

    for line in f.readlines():
        words = line.split()
        if words == []:
            continue

        if words[-1]=='b[k]':
            current_array = 'b'
        elif words[-1]=='c[k]':
            current_array = 'c'
	elif words[-1]=='bhat[k]':
            current_array = 'bhat'
        elif words[-1]=='A[k,j]':
            current_array = 'A'

        try:
            k = int(words[0])
            if current_array == 'b':
                b[k] = float(words[1])
	    elif current_array =='c':
                c[k] = float(words[1])
            elif current_array == 'A':
                j = int(words[1])
                A[k,j] = float(words[2])
        except ValueError:
            continue
 
    return rk.ExplicitRungeKuttaMethod(A=A, b=b, name=name)
