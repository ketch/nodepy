NodePy (Numerical ODEs in Python) is a Python package for designing, analyzing, and testing numerical methods for initial value ODEs. Its development was motivated by my own research in time integration methods for PDEs. I found that I was frequently repeating tasks that could be automated and integrated. Initially I developed a collection of MATLAB scripts, but this became unwieldy due to the large number of files that were necessary and the more limited capability for code reuse.

NodePy represents an object-oriented approach, in which the basic object is a numerical ODE solver. The idea is to design a laboratory for such methods in the same sense that MATLAB is a laboratory for matrices.

Documentation can be found online at

http://web.kaust.edu.sa/faculty/davidketcheson/NodePy/

The development version can be obtained using Mercurial by typing

hg clone https://bitbucket.org/ketch/nodepy



License

NodePy is distributed under the terms of the Berkeley Software Distribution (BSD) license. The license is in the file nodepy/LICENSE.txt and reprinted below.

See http://www.opensource.org/licenses/bsd-license.php for more details.

Copyright (c) 2008-2010 David I. Ketcheson. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        * Neither the name of King Abdullah University of Science & Technology nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Funding

NodePy development has been supported by:

        * A U.S. Dept. of Energy Computational Science Graduate Fellowship
        * Grants from King Abdullah University of Science & Technology


