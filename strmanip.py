def collect_powers(s,v):
    """
        Collect repeated multiplications of string v in s and replace them by exponentiation.

        **Examples**::

            >>> collect_powers('c*c','c')
            'c**2'
            >>> collect_powers('c*c*c*c*c*c*c*c*c*c','c')
            'c**10'
            >>> collect_powers('d*c*c','c')
            'd*c**2'
    """

    import re
    m=s.count(v)
    for i in range(m,0,-1):
        pattern=re.compile(v+'(\*'+v+'){'+str(i)+'}')
        s=pattern.sub(v+'**'+str(i+1),s)
    return s

def getint(string):
    n = int(string[0])
    i = 1
    while True:
        try:
            n = int(string[0:i+1])
        except:
            return n
        i = i + 1

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
    s=s.replace('*1)',')')
    s=s.replace('--','')
    s=s.replace('+-','-')
    s=collect_powers(s,'c')
#    s=collect_sums(s,'c')
#    return str(sympify(s))
    return s


def addper(s):
    return s+'.'

#=====================================================
def get_substring(st,startpos):
#=====================================================
    """
        Extracts everything between an open and close
        brace from a string.

        INPUT:
            st       -- any string (usually a RootedTree)
            startpos -- an integer such that st[startpos] is
                        the open brace ('{') of the desired substring

        OUTPUT:
            A string containing everything from st[startpos] to
            the corresponding close brace ('}') (inclusive).
    """
    return st[startpos:open_to_close(st,startpos)+1]

#=====================================================
def open_to_close(st,startpos):
#=====================================================
    """ 
        Finds end of a substring enclosed by braces starting at startpos.
        Used by get_substring.
    """

    pos=startpos
    openchar=st[pos]
    if openchar=='{':  closechar='}'
    count=1
    while count>0:
        pos+=1
        if st[pos]==openchar:  count+=1
        if st[pos]==closechar: count-=1
    return pos
