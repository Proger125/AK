#include <mpi.h>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
#include <vector>

using namespace std;

const int N = 10;

const double A = 1;
const double B = 1;

const double H = A / (N - 1);
const double L = B / (N - 1);

const double EPS = 0.00001;

double U[N][N];

double map[N];

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

	// Инициализация начальной матрицы

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

	// Разделение области на полосы и распределение по процессам

	int dimension = (N - 2) / size;
	int start = 1 + rank * dimension;
	int end = 1 + (rank + 1) * dimension;

	int* map = new int[N - 1];
	map[0] = 0;
	if (rank == 0) {
		int currentRank = 0;
		for (int i = 1; i < N - 1; i++) {
			map[i] = currentRank;
			if (i % dimension == 0) {
				currentRank++;
			}
		}
	}

	MPI_Bcast(&map[0], N - 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	if (rank == size - 1) {
		dimension += (N - 2) % size;
		start = N - dimension - 1;
		end = N - 1;
	}

	// Алгоритм

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
				U[i][j] = 0.25 * (U[i][j - 1] + U[i][j + 1]
					+ U[i + 1][j] + U[i - 1][j] - (H * H) * f(H * i, L * j));
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

	// Вывод рабочего времени

	if (rank == 0) {
		cout << "Time: " << finish - begin << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	for (int i = 1; i < N - 1; i++) {
		MPI_Bcast(&U[i][0], N, MPI_DOUBLE, map[i], MPI_COMM_WORLD);
	}

	// Вывод данных из нулевого потока в файл
	if (rank == 0) {
		ofstream fout("output.txt");

		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				fout << H * i << " " << H * j << " " << U[i][j] << endl;
			}
		}

		fout.close();
	}
	MPI_Finalize();
}
