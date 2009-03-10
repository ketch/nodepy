import os,sys,pyclaw
x=sys.modules['pyclaw'].__file__
print x
print os.path.dirname(x)
