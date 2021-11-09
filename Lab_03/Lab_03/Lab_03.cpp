#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

#define n 1000

int map[n];
double A[n][n], b[n], c[n], x[n], sum = 0.0;

double* check(double A[n][n], double x[n], double b[n]) {
  double* delta = new double[n];
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += (A[i][j] * x[j]);
    }
    delta[i] = b[i] - sum;
  }
  return delta;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int i, j, k;

  double range = 1.0;
  int rank, nprocs;
  clock_t begin1, end1, begin2, end2;
  MPI_Status status;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


  if (rank == 0) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        A[i][j] = range * (1.0 - 2.0 * (double)rand() / RAND_MAX);
      }
      b[i] = range * (1.0 - 2.0 * (double)rand() / RAND_MAX);
    }

    cout << "Matrix A (generated randomly):" << endl;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        cout << A[i][j] << " ";
      }
      cout << endl;
    }
    cout << "Vector b (generated randomly):" << endl;
    for (i = 0; i < n; i++) {
      cout << b[i] << " ";
    }
    cout << "\n\n";
  }

  begin1 = clock();

  MPI_Bcast(&A[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (i = 0; i < n; i++) {
    map[i] = i % nprocs;
  }

  for (k = 0; k < n; k++) {
    MPI_Bcast(&A[k][k], n - k, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    MPI_Bcast(&b[k], 1, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    for (i = k + 1; i < n; i++) {
      if (map[i] == rank) {
        c[i] = A[i][k] / A[k][k];
      }
    }
    for (i = k + 1; i < n; i++) {
      if (map[i] == rank) {
        for (j = 0; j < n; j++) {
          A[i][j] = A[i][j] - (c[i] * A[k][j]);
        }
        b[i] = b[i] - (c[i] * b[k]);
      }
    }
  }
  end1 = clock();

  if (rank == 0) {
    begin2 = clock();

    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (i = n - 2; i >= 0; i--) {
      sum = 0;

      for (j = i + 1; j < n; j++) {
        sum = sum + A[i][j] * x[j];
      }
      x[i] = (b[i] - sum) / A[i][i];
    }

    end2 = clock();

    cout << "The solution is:" << endl;
    for (i = 0; i < n; i++) {
      printf("\nx%d=%f\t", i, x[i]);
    }


    cout << "MPI Time: " << (double)(end1 - begin1) / CLOCKS_PER_SEC << endl;

    cout << "Back substitution time: "
         << (double)(end2 - begin2) / CLOCKS_PER_SEC << endl;

    cout << "Check:" << endl;
    double* delta = check(A, x, b);
    for (int i = 0; i < n; i++) {
      cout << "Delta " << i << " = " << delta[i] << endl;
    }
  }
  MPI_Finalize();

  return (0);
}