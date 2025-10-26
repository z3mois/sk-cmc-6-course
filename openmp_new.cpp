#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>

inline int idx(int i, int j, int k, int N) {
    return i * (N + 1) * (N + 1) + j * (N + 1) + k;
}

// u = sin(pi x/Lx) * sin(2pi y/Ly) * sin(3pi z/Lz) * cos(at * t)
inline double analytical(double x, double y, double z, double t,
                         double at, double Lx, double Ly, double Lz) {
    return std::sin(M_PI * x / Lx) *
           std::sin(2.0 * M_PI * y / Ly) *
           std::sin(3.0 * M_PI * z / Lz) *
           std::cos(at * t);
}

inline double laplacian(const std::vector<double> &u, int i, int j, int k, int N,
                        double invhx2, double invhy2, double invhz2) {
    double u0 = u[idx(i,j,k,N)];
    double dx = (u[idx(i-1,j,k,N)] - 2.0*u0 + u[idx(i+1,j,k,N)]) * invhx2;
    double dy = (u[idx(i,j-1,k,N)] - 2.0*u0 + u[idx(i,j+1,k,N)]) * invhy2;
    double dz = (u[idx(i,j,k-1,N)] - 2.0*u0 + u[idx(i,j,k+1,N)]) * invhz2;
    return dx + dy + dz;
}

void initialize_u0(std::vector<double> &u0, double at, int N,
                   double hx, double hy, double hz,
                   double Lx, double Ly, double Lz) {
    (void)at; // t=0 => cos(0)=1
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i)
      for (int j = 1; j < N; ++j)
        for (int k = 1; k < N; ++k)
          u0[idx(i,j,k,N)] = analytical(i*hx, j*hy, k*hz, 0.0, at, Lx, Ly, Lz);
}

void initialize_u1(const std::vector<double> &u0, std::vector<double> &u1,
                   int N, double tau,
                   double invhx2, double invhy2, double invhz2) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i)
      for (int j = 1; j < N; ++j)
        for (int k = 1; k < N; ++k)
          u1[idx(i,j,k,N)] = u0[idx(i,j,k,N)]
                           + 0.5 * 0.25 * tau * tau * laplacian(u0,i,j,k,N,invhx2,invhy2,invhz2);
}

// сначала периодика по y, потом Дирихле по x и z
void apply_boundaries(std::vector<double> &u, int N) {
    // periodic Y
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= N; ++i)
      for (int k = 0; k <= N; ++k)
        u[idx(i,0,k,N)] = u[idx(i,N-1,k,N)];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= N; ++i)
      for (int k = 0; k <= N; ++k)
        u[idx(i,N,k,N)] = u[idx(i,1,k,N)];

    // Dirichlet X
    #pragma omp parallel for collapse(2)
    for (int j = 0; j <= N; ++j)
      for (int k = 0; k <= N; ++k) {
        u[idx(0,j,k,N)] = 0.0;
        u[idx(N,j,k,N)] = 0.0;
      }

    // Dirichlet Z
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= N; ++i)
      for (int j = 0; j <= N; ++j) {
        u[idx(i,j,0,N)] = 0.0;
        u[idx(i,j,N,N)] = 0.0;
      }
}

void update(std::vector<double> &unew,
            const std::vector<double> &uold,
            const std::vector<double> &ucur,
            int N, double tau,
            double invhx2, double invhy2, double invhz2) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < N; ++i)
      for (int j = 1; j < N; ++j)
        for (int k = 1; k < N; ++k)
          unew[idx(i,j,k,N)] =
              2.0 * ucur[idx(i,j,k,N)]
            -       uold[idx(i,j,k,N)]
            + 0.25 *(tau * tau) * laplacian(ucur,i,j,k,N,invhx2,invhy2,invhz2);
}

// max-norm
double max_error(const std::vector<double> &u, double t, int N,
                 double hx, double hy, double hz,
                 double Lx, double Ly, double Lz, double at) {
    double m = 0.0;
    #pragma omp parallel for collapse(3) reduction(max:m)
    for (int i = 1; i < N; ++i)
      for (int j = 1; j < N; ++j)
        for (int k = 1; k < N; ++k) {
            double exact = analytical(i*hx, j*hy, k*hz, t, at, Lx, Ly, Lz);
            double e = std::fabs(u[idx(i,j,k,N)] - exact);
            if (e > m) m = e;
        }
    return m;
}




// RMS-ошибка
double rms_error(const std::vector<double> &u, double t, int N,
                 double hx, double hy, double hz,
                 double Lx, double Ly, double Lz, double at) {
    long long cnt = 0;
    double acc = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:acc,cnt)
    for (int i = 1; i < N; ++i)
      for (int j = 1; j < N; ++j)
        for (int k = 1; k < N; ++k) {
            double exact = analytical(i*hx, j*hy, k*hz, t, at, Lx, Ly, Lz);
            double d = u[idx(i,j,k,N)] - exact;
            acc += d * d;
            cnt += 1;
        }
    return std::sqrt(acc / std::max<long long>(1, cnt));
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);

    int N = 128;
    double T = 0.001;
    int Nt = 20;            
    double tau = T / Nt;

    double L = 1.0;
    double Lx = L, Ly = L, Lz = L;

    if (argc > 1) { try { N = (int)std::stod(argv[1]); } catch(...) { std::cerr<<"Arg1=N\n"; return 1; } }
    if (argc > 2) {
        std::string a2 = argv[2];
        if (a2=="pi") { L = M_PI; Lx = Ly = Lz = L; }
        else { try { L = std::stod(a2); Lx = Ly = Lz = L; } catch(...) { std::cerr<<"Arg2=L or 'pi'\n"; return 1; } }
    }
    if (argc > 3) { try { T = std::stod(argv[3]); } catch(...) { std::cerr<<"Arg3=T\n"; return 1; } }
    if (argc > 4) { try { Nt = std::max(1, (int)std::stod(argv[4])); } catch(...) { std::cerr<<"Arg4=Nt\n"; return 1; } }
    tau = T / Nt;

    double hx = Lx / N, hy = Ly / N, hz = Lz / N;
    double invhx2 = 1.0/(hx*hx), invhy2 = 1.0/(hy*hy), invhz2 = 1.0/(hz*hz);
    double a = 0.5;
    //  CFL
    double cfl = a * tau * std::sqrt(invhx2 + invhy2 + invhz2);
    std::cout << "CFL=" << cfl << (cfl <= 1.0 ? " (OK)\n" : " (too high!)\n");

    // a_t = pi / 2* sqrt(1/Lx^2 + 4/Ly^2 + 9/Lz^2)
    double at = M_PI * std::sqrt( (1.0/(Lx*Lx)) + (4.0/(Ly*Ly)) + (9.0/(Lz*Lz)) ) * 0.5;

    std::vector<std::vector<double> > u(3, std::vector<double>((N+1)*(N+1)*(N+1), 0.0));

    double t0 = omp_get_wtime();

    // u^0 и границы
    initialize_u0(u[0], at, N, hx, hy, hz, Lx, Ly, Lz);
    apply_boundaries(u[0], N);

    // u^1 и границы
    initialize_u1(u[0], u[1], N, tau, invhx2, invhy2, invhz2);
    apply_boundaries(u[1], N);

    for (int step = 2; step <= Nt; ++step) {
        update(u[step%3], u[(step+1)%3], u[(step+2)%3],
               N, tau, invhx2, invhy2, invhz2);
        apply_boundaries(u[step%3], N);

        double t = step * tau;
        double e_inf = max_error(u[step%3], t, N, hx, hy, hz, Lx, Ly, Lz, at);
        double e_rms = rms_error(u[step%3], t, N, hx, hy, hz, Lx, Ly, Lz, at);
        double rel = e_inf / 1.0;

        std::cout << "Step " << step << "/" << Nt
                  << " | err_inf=" << std::scientific << e_inf
                  << " | err_rms=" << e_rms
                  << std::fixed
                  << " | rel=" << std::setprecision(4) << (rel*100.0) << "%\n"
                  << std::setprecision(6);
    }

    double t1 = omp_get_wtime();
    std::cout << "Total time: " << (t1 - t0) << " s\n";
    return 0;
}
