
	subroutine solve_fmm(TOL,pot,ltree,npbox,norder,
	1    nboxes,nlevels,itree,iptr,fvals,boxsize,centers)
	implicit real *8 (a-h,o-z)
	integer iptr(9)
	integer npols, nd, ifnear

	integer itree(ltree)
	real *8 boxsize(0:nlevels)
	real *8 fvals(1,npbox,nboxes),centers(2,nboxes)
	real *8 pot(1,npbox,nboxes)
	real *8, allocatable :: xref(:,:)
	real *8 xyztmp(3),rintl(0:200),umat,vmat,wts
	real *8 timeinfo(6),tprecomp(3)

	complex *16 zpars

c       real *8 potmat(np,nboxes)

	
	real *8, allocatable :: potex(:,:,:)
	complex *16 ima,zz,ztmp,zk

	real *8 alpha,beta,targ(2)

	character *1 type
	data ima/(0.0d0,1.0d0)/

	external fgaussn,fgauss1
	logical flag
	
	eps = TOL

	npols = norder*norder
	nd = 1

	zk = ima
	done = 1
	pi = atan(done)*4

	boxlen = 1.0d0

	nd = 1
	
	norder = 8
	iptype = 1
	eta = 1.0d0	
	eps = TOL

	do i=1,nboxes
	   do j=1,npbox
	      pot(1,j,i) = 0
	   enddo
	enddo

	type = 'f'
	ifnear = 1
	iperiod = 0
	
	call lbfmm2d(nd,eps,iperiod,nboxes,nlevels,ltree,
	1    itree,iptr,norder,npols,fvals,centers,boxsize,
	2    ifnear,pot,timeinfo)
	
	end
