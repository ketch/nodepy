def collect_powers(s,v):
    import re
    m=s.count(v)
    for i in range(m,0,-1):
        pattern=re.compile(v+'(\*'+v+'){'+str(i)+'}')
        s=pattern.sub(v+'**'+str(i+1),s)
    return s

def collect_sums(s,v):
    import re
    m=s.count(v)
    for i in range(m,1,-1):
        pattern=re.compile('(\+'+v+'){'+str(i)+'}')
        s=pattern.sub(str(i)+'*'+v,s)
    return s

def mysimp(s):
    from sympy import sympify
    s=s.replace('1*','')
    s=s.replace('*1','')
    s=s.replace('--','')
    s=s.replace('+-','-')
    s=collect_powers(s,'c')
#    s=collect_sums(s,'c')
    return str(sympify(s))

def addper(s):
    return s+'.'


