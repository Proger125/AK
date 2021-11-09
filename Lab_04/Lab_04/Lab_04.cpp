#include <mpi.h>
#include <cmath>
#include <iostream>
#include <time.h>

using namespace std;

const int N = 4;

const double A = 1;
const double B = 1;

const double H = A / (N - 1);
const double L = B / (N - 1);

const double EPS = 0.001;

double U[N][N];

double f(double x, double y) { return y * x; }

double f1(double x, double y) { return y * y; }

double f2(double x, double y) { return y * y * y; }

double f3(double x, double y) { return x * x * x; }

double f4(double x, double y) { return x * x; }

int main(int argc, char** argv) {
  int size;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        U[i][j] = 0;
      }
    }

    for (int i = 0; i < N; i++) {
      U[0][i] = f1(0, L * i);
      U[N - 1][i] = f2(A, L * i);
      U[i][0] = f3(H * i, 0);
      U[i][N - 1] = f4(H * i, B);
    }
  }

  MPI_Bcast(&U[0][0], N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int dimension = (N - 2) / size;
  int start = 1 + rank * dimension;
  int end = 1 + (rank + 1) * dimension;

  if (rank == size - 1) {
    dimension += N % size;
    start = N - dimension - 1;
    end = N - 1;
  }

  //cout << dimension << endl;

  clock_t begin = clock();

  double maxDelta;
  double delta;
  do {
    if (rank != size - 1) {
      MPI_Send(&U[end - 1][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank != 0) {
      MPI_Recv(&U[start - 1][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    if (rank != 0) {
      MPI_Send(&U[start][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    }

    if (rank != size - 1) {
      MPI_Recv(&U[end][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    delta = 0;
    for (int i = start; i < end; i++) {
      for (int j = 1; j < N - 1; j++) {
        double temp = U[i][j];
        U[i][j] = ((U[i][j - 1] + U[i][j + 1]) * (H * H)
            + (U[i + 1][j] + U[i - 1][j]) * (L * L) - (H * L) * f(H * i, L * j));
        double currentDelta = abs(temp - U[i][j]);
        if (currentDelta > delta) {
          delta = currentDelta;
        }
      }
    }

    MPI_Reduce(&delta, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxDelta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      cout << "Current delta = " << maxDelta << endl;
    }

  } while (maxDelta > EPS);

  MPI_Barrier(MPI_COMM_WORLD);

  clock_t finish = clock();

  MPI_Finalize();
}
