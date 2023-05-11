#gfortran -shared -fPIC solve_fmm.o -w -o int2.so ${HOME}/lib/libboxcode2dlegacy.so -fopenmp -lblas -llapack

FCOMP=gfortran

LIBNAME=juliawraplibboxcode2dlegacy

DYNAMICLIB = $(LIBNAME).so

DYLIBS_INSTALL_DIR = ${HOME}/lib

LBLAS =  -lblas -llapack

DYLIBS =  ${DYLIBS_INSTALL_DIR}/libboxcode2dlegacy.so

FFLAGS = -fPIC -O3 -march=native -funroll-loops -ftree-vectorize -ffast-math -std=legacy -c -w

DYLIBSFLAGS = -fopenmp $(LBLAS)

OBJS = ./solve_fmm.o

%.o: %.f
	$(FCOMP) $(FFLAGS) -o $@ $^
lib: $(DYNAMICLIB)

$(DYNAMICLIB): $(OBJS) 
	$(FCOMP) -shared -fPIC $(OBJS) -w -o $(DYNAMICLIB) $(DYLIBS) $(DYLIBSFLAGS)

clean:
	rm solve_fmm.o
	rm juliawraplibboxcode2dlegacy.so
