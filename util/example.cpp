# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <random>

# include <fftw3.h>

int main ( void );
void test01 ( void );
void test02 ( void );
double frand ( void ){
	return rand()%1000;
};
const double pi = 3.1415927;

/******************************************************************************/

int main ( void )

/******************************************************************************/
/*
  Purpose:

    FFTW3_PRB demonstrates the use of FFTW3.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    05 November 2007

  Author:

    John Burkardt
*/
{

  printf ( "\n" );
  printf ( "FFTW3_PRB\n" );
  printf ( "  C version\n" );
  printf ( "  Test the FFTW3 library.\n" );

  test01 ( );
 // test02 ( );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "FFTW3_PRB\n" );
  printf ( "  Normal end of execution.\n" );
 
  printf ( "\n" );

  return 0;
}
/******************************************************************************/

void test01 ( void )

/******************************************************************************/
/*
  Purpose:

    TEST01: apply FFT to complex 1D data.

  Discussion:

    In this example, we generate N=100 random complex values stored as
    a vector of type FFTW_COMPLEX named "IN".

    We have FFTW3 compute the Fourier transform of this data named "OUT".

    We have FFTW3 compute the inverse Fourier transform of "OUT" to get
    "IN2", which should be the original input data, scaled by N.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 November 2007

  Author:

    John Burkardt
*/
{
  int i;
  fftw_complex *in;
  fftw_complex *in2;
  int n = 100;
  fftw_complex *out;
  fftw_plan plan_backward;
  fftw_plan plan_forward;
  unsigned int seed = 123456789;

  printf ( "\n" );
  printf ( "TEST01\n" );
  printf ( "  Demonstrate FFTW3 on a single vector of complex data.\n" );
  printf ( "\n" );
  printf ( "  Transform data to FFT coefficients.\n" );
  printf ( "  Backtransform FFT coefficients to recover data.\n" );
  printf ( "  Compare recovered data to original data.\n" );
/*
  Create the input array.
*/
  in = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * n );

  srand ( seed );

  for ( i = 0; i < n; i++ )
  {
    in[i][0] = sin(double(i)*pi/20);//+frand ( ) ;
    in[i][1] = cos(double(i)*pi/20);//frand ( );
  }

  printf ( "\n" );
  printf ( "  Input Data:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %3d  %12f  %12f\n", i, in[i][0], in[i][1] );
  }
/*
  Create the output array.
*/
  out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * n );

  plan_forward = fftw_plan_dft_1d ( n, in, out, FFTW_FORWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_forward );

  printf ( "\n" );
  printf ( "  Output FFT Coefficients:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %3d  %12f  %12f\n", i, out[i][0], out[i][1] );
  }
/*
  Recreate the input array.
*/
  in2 = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * n );

  plan_backward = fftw_plan_dft_1d ( n, out, in2, FFTW_BACKWARD, FFTW_ESTIMATE );

  fftw_execute ( plan_backward );

  printf ( "\n" );
  printf ( "  Recovered input data:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %3d  %12f  %12f\n", i, in2[i][0], in2[i][1] );
  }

  printf ( "\n" );
  printf ( "  Recovered input data divided by N:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %3d  %12f  %12f\n", i, 
      in2[i][0] / ( double ) ( n ), in2[i][1] / ( double ) ( n ) );
  }
/*
  Free up the allocated memory.
*/
  fftw_destroy_plan ( plan_forward );
  fftw_destroy_plan ( plan_backward );

  fftw_free ( in );
  fftw_free ( in2 );
  fftw_free ( out );

  return;
}
/******************************************************************************/

void test02 ( void )

/******************************************************************************/
/*
  Purpose:

    TEST02: apply FFT to real 1D data.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    23 October 2005

  Author:

    John Burkardt
*/
{
  int i;
  double *in;
  double *in2;
  int n = 100;
  int nc;
  fftw_complex *out;
  fftw_plan plan_backward;
  fftw_plan plan_forward;
  unsigned int seed = 123456789;

  printf ( "\n" );
  printf ( "TEST02\n" );
  printf ( "  Demonstrate FFTW3 on a single vector of real data.\n" );
  printf ( "\n" );
  printf ( "  Transform data to FFT coefficients.\n" );
  printf ( "  Backtransform FFT coefficients to recover data.\n" );
  printf ( "  Compare recovered data to original data.\n" );
/*
  Set up an array to hold the data, and assign the data.
*/
  in = (double*) fftw_malloc ( sizeof ( double ) * n );

  srand ( seed );

  for ( i = 0; i < n; i++ )
  {
    in[i] = frand ( );
  }

  printf ( "\n" );
  printf ( "  Input Data:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %4d  %12f\n", i, in[i] );
  }
/*
  Set up an array to hold the transformed data,
  get a "plan", and execute the plan to transform the IN data to
  the OUT FFT coefficients.
*/
  nc = ( n / 2 ) + 1;

  out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * nc );

  plan_forward = fftw_plan_dft_r2c_1d ( n, in, out, FFTW_ESTIMATE );

  fftw_execute ( plan_forward );

  printf ( "\n" );
  printf ( "  Output FFT Coefficients:\n" );
  printf ( "\n" );

  for ( i = 0; i < nc; i++ )
  {
    printf ( "  %4d  %12f  %12f\n", i, out[i][0], out[i][1] );
  }
/*
  Set up an arrray to hold the backtransformed data IN2,
  get a "plan", and execute the plan to backtransform the OUT
  FFT coefficients to IN2.
*/
  in2 = (double*) fftw_malloc ( sizeof ( double ) * n );

  plan_backward = fftw_plan_dft_c2r_1d ( n, out, in2, FFTW_ESTIMATE );

  fftw_execute ( plan_backward );

  printf ( "\n" );
  printf ( "  Recovered input data divided by N:\n" );
  printf ( "\n" );

  for ( i = 0; i < n; i++ )
  {
    printf ( "  %4d  %12f\n", i, in2[i] / ( double ) ( n ) );
  }
/*
  Release the memory associated with the plans.
*/
  fftw_destroy_plan ( plan_forward );
  fftw_destroy_plan ( plan_backward );

  fftw_free ( in );
  fftw_free ( in2 );
  fftw_free ( out );

  return;
}
