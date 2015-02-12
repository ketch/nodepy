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
    for key,method in methods.iteritems():
        fname = method.shortname+'.html'
        print fname
        #write_page(method, fname=fname)
        write_method_page(method, fname=fname)

    write_index_page(methods)


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
    stabfun = "$$"+rational_function_latex(p,q)+"$$"

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


def write_index_page(methods):
    """
    Write a very plain page with HTML links to all the pages
    for specific methods.

    TODO:

        - Make this page nicer
    """
    outfile = open('index.html','w')

    outfile.write('<head>\n')
    write_mathjax_header(outfile)

    outfile.write('\n<title>Numipedia</title>\n')
    outfile.write('</head>\n')

    outfile.write('<h1>Numipedia: an encyclopedia of numerical integrators</h1>\n')

    for key,method in methods.iteritems():
        fname = method.shortname+'.html'
        outfile.write('<a href="'+fname+'">'+method.name+'</a><br>\n')
    outfile.close()
 

def poly1d_latex(p):
    """Format a numpy poly1d object in LaTeX."""
    c = p.coeffs[::-1]
    s = str(c[0])
    if len(c)>1:
        x = str_add_coeff(c[1])
        if x:
            s += x + " z "
    for i,coeff in enumerate(c[2:]):
        x = str_add_coeff(c[i+2])
        if x:
            s += x + " z^{"+str(i+2)+"} "
    return s

def str_add_coeff(c):
    from sympy import latex
    if c == 0:
        return False
    elif c == 1:
        return ' + '
    elif c == -1:
        return ' - '
    elif c>0:
        return ' + '+latex(c)
    else: # c<0 
        return latex(c)

def rational_function_latex(p,q):
    """Format rational function p(z)/q(z) in laTeX."""
    if len(q.coeffs)==1:
        return poly1d_latex(p)
    else:
        return r"\frac{"+poly1d_latex(p)+"}{"+poly1d_latex(q)+"}"

def write_mathjax_header(outfile):
    outfile.write(r"""<script type="text/x-mathjax-config">
                     MathJax.Hub.Config({
                       tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
                     });
                     </script>""")
    outfile.write('\n')
    #outfile.write('<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>')
    outfile.write('<script type="text/javascript" src="file://Users/ketch/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>')

def tex(s):
    return "$"+str(s)+"$"


def write_page(method,fname='test.html'):
    """
    Write a HTML page for a method from scratch (no template).
    Currently not used.
    """
    outfile = open(fname,'w')

    outfile.write('<head>\n')
    write_mathjax_header(outfile)

    outfile.write('<title>')
    outfile.write(method.name)
    outfile.write('</title>\n')

    outfile.write('</head>\n')

    outfile.write('<a href="index.html">Back to index</a>\n')

    outfile.write('<h1>'+method.name+'</h1>\n')
    outfile.write('<h2>'+method.info+'</h2>\n')
    # Need to switch to terminal font here
    outfile.write('<blockquote>\n')
    outfile.write('<div>\n')
    outfile.write(method.latex())
    outfile.write('</div>\n')
    outfile.write('</blockquote>\n')
    outfile.write('<br><br>\n\n')

    #plot_file = 'img/'+method.shortname+'.png'
    plot_file = method.shortname+'.png'
    method.plot_stability_region(to_file=plot_file)
    outfile.write('<img src='+plot_file+' height="400" alt="Absolute stability region" />')

    outfile.write('<br><br>\n\n')
    # Need to format this better:
    outfile.write('Stability polynomial: '+poly1d_latex(method.stability_function()[0]))
    outfile.write('<br><br>\n\n')
    outfile.write('Radius of absolute monotonicity: '+str(method.absolute_monotonicity_radius()))

    outfile.close()


if __name__ == "__main__":
    write_numipedia()
