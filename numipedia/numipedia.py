"""
Generate an encyclopedia of numerical methods, using NodePy.
The encyclopedia may include:

    - A page for each method, with its coefficients and properties listed and plotted
    - A system of automatically generated tags, with an index page of tags and a page for each tag listing all so-tagged methods
    - More advanced: the ability to generate method-to-method comparisons
"""
def write_numipedia():
    from nodepy import rk
    methods = rk.loadRKM('All')
    for key,method in methods.iteritems():
        fname = method.shortname+'.html'
        print fname
        #write_page(method, fname=fname)
        write_page_from_template(method, fname=fname)

    write_index_page(methods)

def write_page_from_template(method,fname='test.html',template_file='method_template.html'):
    from mako.template import Template
    import matplotlib.pyplot as plt

    plot_file = method.shortname+'.pdf'
    fig = method.plot_stability_region(to_file=plot_file,longtitle=False)
    plt.close()
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
    outfile = open('index.html','w')

    outfile.write('<head>\n')
    write_mathjax_header(outfile)

    outfile.write('<title>')
    outfile.write('Numipedia')
    outfile.write('</title>\n')

    outfile.write('</head>\n')

    outfile.write('<h1>Numipedia: an encyclopedia of numerical integrators</h1>\n')

    for key,method in methods.iteritems():
        fname = method.shortname+'.html'
        outfile.write('<a href="'+fname+'">'+method.name+'</a><br>\n')
    outfile.close()
 
def write_page(method,fname='test.html'):
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

def poly1d_latex(p):
    """Format numpy poly1d object in LaTeX."""
    c = p.coeffs[::-1]
    s = str(c[0])
    if len(c)>1:
        x = str_add_coeff(c[1])
        if x:
            s += x + " z "
    for i,coeff in enumerate(c[2:]):
        x = str_add_coeff(c[1])
        if x:
            s += x + " z^{"+str(i+2)+"} "
    return s

def str_add_coeff(c):
    if c == 0:
        return False
    elif c == 1:
        return ' + '
    elif c == -1:
        return ' - '
    elif c>0:
        return ' + '+str(c)
    else: # c<0 
        return str(c)

def rational_function_latex(p,q):
    """Format p/q in laTeX."""
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


if __name__ == "__main__":
    write_numipedia()
