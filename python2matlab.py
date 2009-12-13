def python_to_matlab(code):
    r"""
        Convert python code string (order condition) to matlab code string
        Doesn't really work yet.  We need to do more parsing.
    """
    print code
    outline=code
    outline=outline.replace("**",".^")
    outline=outline.replace("*",".*")
    outline=outline.replace("dot(b,","b'*(")
    outline=outline.replace("dot(bhat,","bhat'*(")
    outline=outline.replace("dot(Ahat,","Ahat*(")
    outline=outline.replace("dot(A,","(A*(")
    outline=outline.replace("( c)","c")
    outline=outline.replace("-0","")
    print outline
    print '******************'
    return outline
