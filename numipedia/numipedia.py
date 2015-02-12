"""
Generate an encyclopedia of numerical methods, using NodePy.
The encyclopedia may include:

    - A page for each method, with its coefficients and properties listed and plotted
    - A system of automatically generated tags, with an index page of tags and a page for each tag listing all so-tagged methods
    - More advanced: the ability to generate method-to-method comparisons
    - The ability to search for all methods satisfying a given set of criteria
"""
def write_numipedia():
    """
    Main function to write the whole numipedia site.
    Loops over all methods in rk.loadRKM('All') and writes a page for each.

    TODO:

        - Add parameterized method families
        - Add more methods
    """
    from nodepy import rk
    methods = rk.loadRKM('All')

    write_index_page(methods)

    for key,method in methods.iteritems():
        fname = method.shortname+'.html'
        print fname
        write_method_page(method, fname=fname)



def write_method_page(method,fname='test.html',template_file='method_template.html'):
    """
    Writes an HTML page for a specified method, using the template file.
    Uses mako to do simple substitutions.

    TODO:

        - Add more method properties
        - Improve page layout (in template)
    """
    from mako.template import Template
    import matplotlib.pyplot as plt

    # Plot stability region.
    plot_file = method.shortname+'.png'
    fig = method.plot_stability_region(to_file=plot_file,longtitle=False)
    plt.close()

    # Compute and render stability function
    p,q=method.stability_function()
    from sympy import symbols, latex
    z = symbols('z')
    pp = sum(co*z**i for i,co in enumerate(p.c[::-1]))
    qq = sum(co*z**i for i,co in enumerate(q.c[::-1]))
    stabfun = "$$"+latex(pp/qq,order='old')+"$$"

    if method.info is '':
        method.info = method.mtype

    mytemplate = Template(filename=template_file)
    s = mytemplate.render(name=method.name,
                          desc=method.info,
                          plot_file=plot_file,
                          butcher=method.latex(),
                          stabfun=stabfun,
                          amrad=tex(method.absolute_monotonicity_radius()),
                          order = tex(method.__num__().order()),
                          stage_order = tex(method.__num__().stage_order())
                          )

    outfile = open(fname,'w')
    outfile.write(s)
    outfile.close()


def write_index_page(methods,fname='index.html',template_file='index_template.html'):
    from mako.template import Template
    mytemplate = Template(filename = template_file)

    class_str = {}
    for method_name, method in methods.iteritems():
        properties = ["method"]

        if method.is_explicit(): properties.append("explicit")
        else: properties.append("implicit")
        if method.mtype == 'Diagonally implicit Runge-Kutta method':
            properties.append('diagonally-implicit')

        if method.absolute_monotonicity_radius()>1.e-10:
            properties.append("ssp")
        else: properties.append("not-ssp")

        properties.append("order-%s" % str(method.p))

        if hasattr(method,'embedded_method'):
            properties.append("pair")
        else:
            properties.append("not-pair")

        class_str[method.shortname] = " ".join(properties)

    s = mytemplate.render(methods=methods,
                          class_str=class_str)
    with open(fname,'w') as outfile:
        outfile.write(s)

def tex(s):
    from nodepy.snp import printable
    return "$"+printable(s,return_zero=True)+"$"

if __name__ == "__main__":
    write_numipedia()
