Integral equation solver for Poisson's equation in the plane.

The following has been verified on:
1, Fedora Linux 37 (Workstation Edition) x86_64
2, Ubuntu 20.04.6 LTS x86_64

Install:

1, Download the 2d boxcode for an 8th order volume fmm: https://github.com/mrachh/boxcode2d-legacy
2, Follow the install instructions for the boxcode
3, Run make install from poisson2d/src/fortan
4, Launch the Julia REPL in poisson2d/
4, In the Julia REPL, activate the project through entering ']' followed by 'instantiate'

Run the code:

1, Launch the Julia REPL
2, In the Julia REPL, write ']' followed by 'activate .'
3, In the Julia REPL, write 'include("startup.jl")'
