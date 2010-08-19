def bisect(rlo, rhi, acc, tol, fun, params=None):
    """ 
        Performs a bisection search.

        **Input**:
            - fun -- a function such that fun(r)==True iff x_0>r, where x_0 is the value to be found.
    """
    while rhi-rlo>acc:
        r=0.5*(rhi+rlo)
        if params: isvalid=fun(r,tol,params)
        else: isvalid=fun(r,tol)
        if isvalid:
            rlo=r
        else:
            rhi=r
    return rlo

def permutations(str):
    if len(str) <=1:
        yield str
    else:
        for perm in permutations(str[1:]):
            for i in range(len(perm)+1):
                yield perm[:i] + str[0:1] + perm[i:]
