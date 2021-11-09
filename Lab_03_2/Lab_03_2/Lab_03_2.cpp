#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

using namespace std;

#define n 10

int map[n];
double A[n][n];  // Matrix
double B[n][n];  // Inverse matrix
double C[n][n];  // Help matrix
double c[n];
double coef;

int main(int argc, char** argv) {
  srand(time(NULL));
  MPI_Init(&argc, &argv);

  int i, j, k;

  double range = 1.0;

  int rank, nprocs;
  clock_t begin1, end1, begin2, end2;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        //A[i][j] = range * (1.0 - 2.0 * (double)rand() / RAND_MAX);
        A[i][j] = rand() % n + 1;
        if (i == j) {
          B[i][j] = 1;
          C[i][j] = 1;
        } else {
          B[i][j] = 0;
          C[i][j] = 0;
        }
      }
    }
    /*
    A[0][0] = 1;
    A[0][1] = 2;
    A[1][0] = 3;
    A[1][1] = 4;
    */
    /*
      A[0][0] = -1;
      A[0][1] = 2;
      A[0][2] = -2;

      A[1][0] = 2;
      A[1][1] = -1;
      A[1][2] = 5;

      A[2][0] = 3;
      A[2][1] = -2;
      A[2][2] = 4;
    */

    cout << "Matrix A (generated randomly):" << endl;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        cout << A[i][j] << " ";
      }
      cout << endl;
    }
  }

  MPI_Bcast(&A[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&C[0][0], n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (i = 0; i < n; i++) {
    map[i] = i % nprocs;
  }
  // Straight run
  begin1 = clock();

  for (k = 0; k < n; k++) {
    if (map[k] == rank) {
      coef = A[k][k];
      for (i = 0; i < n; i++) {
        A[k][i] = A[k][i] / coef;
        B[k][i] = B[k][i] / coef;
      }
    }
    MPI_Bcast(&A[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    MPI_Bcast(&B[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);

    for (i = k + 1; i < n; i++) {
      if (map[i] == rank) {
        c[i] = A[i][k] / A[k][k];
        for (j = 0; j < n; j++) {
          A[i][j] = A[i][j] - (c[i] * A[k][j]);
          B[i][j] = B[i][j] - (c[i] * B[k][j]);
        }
      }
    }
  }

  end1 = clock();

  // Reverse run
  begin2 = clock();

  for (k = n - 1; k >= 0; k--) {
    MPI_Bcast(&A[k][k], n - k, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    MPI_Bcast(&B[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    for (i = k - 1; i >= 0; i--) {
      if (map[i] == rank) {
        c[i] = A[i][k] / A[k][k];
        for (j = 0; j < n; j++) {
          A[i][j] = A[i][j] - (c[i] * A[k][j]);
          B[i][j] = B[i][j] - (c[i] * B[k][j]);
        }
      }
    }
  }

  end2 = clock();

  if (rank == 0) {
    cout << "The inverse matrix is:" << endl;
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        cout << B[i][j] << " ";
      }
      cout << endl;
    }
    cout << "Straight run time: " << end1 - begin1 << endl;
    cout << "Reverse run time: " << end2 - begin2 << endl;
    cout << "Whole algorithm time: " << end2 - begin1 << endl;
  }

  // Check
  /*
  for (k = 0; k < n; k++) {
    if (map[k] == rank) {
      coef = B[k][k];
      for (i = 0; i < n; i++) {
        B[k][i] = B[k][i] / coef;
        C[k][i] = C[k][i] / coef;
      }
    }
    MPI_Bcast(&B[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    MPI_Bcast(&C[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);

    for (i = k + 1; i < n; i++) {
      if (map[i] == rank) {
        c[i] = B[i][k] / B[k][k];
        for (j = 0; j < n; j++) {
          B[i][j] = B[i][j] - (c[i] * B[k][j]);
          C[i][j] = C[i][j] - (c[i] * C[k][j]);
        }
      }
    }
  }

  for (k = n - 1; k >= 0; k--) {
    MPI_Bcast(&B[k][k], n - k, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    MPI_Bcast(&C[k][0], n, MPI_DOUBLE, map[k], MPI_COMM_WORLD);
    for (i = k - 1; i >= 0; i--) {
      if (map[i] == rank) {
        c[i] = B[i][k] / B[k][k];
        for (j = 0; j < n; j++) {
          B[i][j] = B[i][j] - (c[i] * B[k][j]);
          C[i][j] = C[i][j] - (c[i] * C[k][j]);
        }
      }
    }
  }

  end2 = clock();
  */
  MPI_Finalize();

  return (0);
}