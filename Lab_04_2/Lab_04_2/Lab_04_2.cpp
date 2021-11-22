#include <iostream>
#include <mpi.h>
#include <time.h>
#include <fstream>

using namespace std;

const int N = 100;

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

int main(int argc, char** argv)
{
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

	// Разделение области на прямоугольники и распределение по процессам

	int r;
	int s;

	// Костыль для 1 потока
	if (size == 1) {
		r = 0;
		s = 1;
	}
	else {
		r = rank % (size / 2);
		s = size / 2;
	}

	int dimension = (N - 2) / size;
	int row = rank > s - 1 ? 1 : 0;

	int start_x = 1 + r * dimension;
	int start_y = 1 + dimension * row;

	int end_x = 1 + (r + 1) * dimension;
	int end_y = 1 + dimension * (row + 1);

	int map[N - 1][N - 1];
	if (rank == 0) {
		int currentRank = 0;
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				map[i][j] = currentRank;
			}
			if (i % dimension == 0 && currentRank != size - 1) {
				currentRank++;
			}
		}
	}


	MPI_Bcast(&map[0][0], (N - 1) * (N - 1), MPI_INT, 0, MPI_COMM_WORLD);

	// Создание типа данных столбец

	MPI_Datatype dt_column;
	MPI_Type_vector(N, 1, N, MPI_DOUBLE, &dt_column);
	MPI_Type_commit(&dt_column);

	if (r == s - 1) {
		end_x = N - 1;
	}

	if (rank > s - 1) {
		end_y = N - 1;
	}

	// Алгоритм

	clock_t begin = clock();

	double maxDelta;
	double delta;
	do {

		if (r != 0) {
			MPI_Send(&U[start_y][start_x], 1, dt_column, rank - 1, 0, MPI_COMM_WORLD);
		}

		if (r != s - 1) {
			MPI_Recv(&U[start_y][end_x], 1, dt_column, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if (rank < size / 2) {
			MPI_Send(&U[end_y - 1][start_x], end_x - start_x, MPI_DOUBLE, rank + (size / 2), 0, MPI_COMM_WORLD);
		}

		if (rank >= s) {
			MPI_Recv(&U[start_y - 1][start_x], end_x - start_x, MPI_DOUBLE, rank - (size / 2), 0, MPI_COMM_WORLD,
				MPI_STATUSES_IGNORE);
		}

		if (r != s - 1) {
			MPI_Send(&U[start_y][end_x - 1], 1, dt_column, rank + 1, 0, MPI_COMM_WORLD);
		}

		if (r != 0) {
			MPI_Recv(&U[start_y][start_x - 1], 1, dt_column, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if (rank >= s) {
			MPI_Send(&U[start_y][start_x], end_x - start_x, MPI_DOUBLE, rank - (size / 2), 0, MPI_COMM_WORLD);
		}

		if (rank < size / 2) {
			MPI_Recv(&U[end_y][start_x], end_x - start_x, MPI_DOUBLE, rank + (size / 2), 0, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
		}

		delta = 0;
		for (int i = start_y; i < end_y; i++) {
			for (int j = start_x; j < end_x; j++) {
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

	} while (maxDelta > EPS);

	clock_t finish = clock();

	MPI_Barrier(MPI_COMM_WORLD);

	// Вывод рабочего времени

	if (rank == 0) {
		cout << "Time: " << finish - begin << endl;
	}

	// Сбор результатов в нулевой поток

	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			MPI_Bcast(&U[i][j], 1, MPI_DOUBLE, map[i][j], MPI_COMM_WORLD);
		}
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
