def ncfun(t,r,x):
    from numpy import exp

    alpha=20.
    q=1.
    p=.1
    t_crit=15
    tau=5.9314;

    dx=x[1]-x[0]
    integral=0.5*(r[0]+r[-1])+r[1:-1].sum()
    integral*=dx
    Ie=t<t_crit
    rdot=-r+1./(1.+exp(-alpha*(-0.5-q*x+p*r+q*integral+Ie)))
    return rdot/tau
