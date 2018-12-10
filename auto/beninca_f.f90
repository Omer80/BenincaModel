!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
!   A periodically forced Beninca model DOI: 10.1073/pnas.1421968112
!------------------------------------------------------------------------------
!------------------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP) 
!     ---------- ---- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
      DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

      DOUBLE PRECISION mB,muB,cBR,mA,muAR,muAB,cAR,cAB,mM,muM
      DOUBLE PRECISION cM,alpha,Tmean,Tmax,freq
      DOUBLE PRECISION TT,PI,TPI,Ft,R,SS
      DOUBLE PRECISION B0,BA,A,M,X,Y
      PI=4.D0*DATAN(1.D0)
      

      mB   = PAR(1)
      muB  = PAR(2)
      cBR  = PAR(3)
      mA   = PAR(4)
      muAR = PAR(5)
      muAB = PAR(6)
      cAR  = PAR(7)
      cAB  = PAR(8)
      mM   = PAR(9)
      muM  = PAR(13)
      cM   = PAR(14)
      alpha= PAR(15)
      Tmean= PAR(16)
      Tmax = PAR(17)
      TT   = PAR(11)
      freq = PAR(18)
      
      B0 = U(1)
      BA = U(2)
      A  = U(3)
      M  = U(4)
      X  = U(5)
      Y  = U(6)
      
      SS = X**2 + Y**2
!      Ft=(1.0+alpha*(Tmax-Tmean)*symcos(2.0*PI*(t-32.0)/365.0))
      Ft=(1.0+alpha*(Tmax-Tmean)*X)
      R = 1.0 - B0 - A - M
      
      F(1) = cBR*(B0+BA)*R-cAB*A*B0-cM*M*B0-mB*B0+Ft*mA*BA
      F(2) = cAB*A*B0-cM*M*BA-mB*BA-Ft*mA*BA
      F(3) = cAR*A*R+cAB*A*B0-cM*M*A-Ft*mA*A
      F(4) = cM*M*(B0+A)-Ft*mM*M
      F(5) = X + (2*PI*freq)*Y - X*SS
      F(6) = -(2*PI*freq)*X + Y - Y*SS
       
      END SUBROUTINE FUNC

      SUBROUTINE STPNT(NDIM,U,PAR,T)  
!     ---------- -----

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: T

      DOUBLE PRECISION mB,muB,cBR,mA,muAR,muAB,cAR,cAB,mM,muM
      DOUBLE PRECISION cM,alpha,Tmean,Tmax,freq
      DOUBLE PRECISION TT,PI,TPI
      DOUBLE PRECISION B0,BA,A,M,X,Y
      INTEGER j
      
!      open(unit = 20, file = 'tlm_parameters.txt', status = 'old', action = 'read')
!      do j = 1,  20
!        read(20,*) test
!        print *, 'N1=', test
!      end do
      PI=4.D0*DATAN(1.D0)
!      print *, 'PI=', PI
!     DOI: 10.1073/pnas.1421968112
      mB   = 0.003
      muB  = 0.015
      cBR  = 0.018
      mA   = 0.013
      muAR = 0.008
      muAB = 0.036
      cAR  = 0.021
      cAB  = 0.049
      mM   = 0.017
      muM  = 0.061
      cM   = 0.078
      alpha= 0.0
      Tmean= 17.1
!      Tmax = 20.5
      Tmax = 17.1
      freq  = 1.0/365.0
      print *, 'freq=', freq
      
      TPI=8*ATAN(1.0D0)
      
      PAR(1) = mB
      PAR(2) = muB
      PAR(3) = cBR
      PAR(4) = mA
      PAR(5) = muAR
      PAR(6) = muAB
      PAR(7) = cAR
      PAR(8) = cAB
      PAR(9) = mM
      PAR(13)= muM
      PAR(14)= cM
      PAR(15)= alpha
      PAR(16)= Tmean
      PAR(17)= Tmax
      PAR(18)= freq
      PAR(11)= TPI/freq
      
      U(1)= 0.16326531
      U(2)= 0.01749869
      U(3)= 0.05468341
      U(4)= 0.11538462
      U(5)=SIN(TPI*T-32/365)
      U(6)=COS(TPI*T-32/365)
       
      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND
      END SUBROUTINE ICND

      SUBROUTINE FOPT
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
