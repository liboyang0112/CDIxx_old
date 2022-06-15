#include <fftw3.h>
int main(){
	fftw_complex *in, *out;
	fftw_plan p;
	int N = 10;
	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
	p = fftw_plan_dft_2d(N, N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_print_plan(p);
	fftw_execute(p); /* repeat as needed */
	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
}
