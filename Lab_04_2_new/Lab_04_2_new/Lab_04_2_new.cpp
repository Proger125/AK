#include <iostream>
#include <mpi.h>
#include <time.h>
#include <fstream>
#include <math.h>

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

	int dims[2] = { 0, 0 };
	MPI_Dims_create(size, 2, dims);

	int periods[2] = { false, false };

	int reorder = true;

	MPI_Comm new_com;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, false, &new_com);
	MPI_Comm_rank(new_com, &rank);

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

	MPI_Bcast(&U[0][0], N * N, MPI_DOUBLE, 0, new_com);

	int left_neighbor;
	int right_neighbor;
	int top_neighbor;
	int bottom_neighbor;

	MPI_Cart_shift(new_com, 1, 1, &left_neighbor, &right_neighbor);
	MPI_Cart_shift(new_com, 0, 1, &bottom_neighbor, &top_neighbor);

	int coords[2];
	MPI_Cart_coords(new_com, rank, 2, coords);

	int amount_x;
	int amount_y;

	MPI_Reduce(&coords[0], &amount_y, 1, MPI_INT, MPI_MAX, 0, new_com);
	MPI_Bcast(&amount_y, 1, MPI_INT, 0, new_com);

	MPI_Reduce(&coords[1], &amount_x, 1, MPI_INT, MPI_MAX, 0, new_com);
	MPI_Bcast(&amount_x, 1, MPI_INT, 0, new_com);

	amount_x++;
	amount_y++;

	if (rank == 0) {
		cout << amount_y << " " << amount_x << endl;
	}

	int dimension_x = (N - 2) / amount_x;
	int dimension_y = (N - 2) / amount_y;

	int start_x = coords[1] * dimension_x + 1;
	int start_y = coords[0] * dimension_y + 1;

	int end_x = start_x + dimension_x;
	int end_y = start_y + dimension_y;

	if (coords[0] == amount_y - 1) {
		end_y = N - 1;
	}

	if (coords[1] == amount_x - 1) {
		end_x = N - 1;
	}

	if (rank == 0) {
		cout << dimension_y << " " << dimension_y << endl;
	}
	int map[N - 1][N - 1];
	if (rank == 0) {
		int currentRank = 0;
		int t = 0;
		for (int i = 1; i < N - 1; i++) {
			if (i % (dimension_y + 1) == 0) {
				currentRank += t + 1;
			}
			int temp = 0;
			for (int j = 1; j < N - 1; j++) {
				if (j % (dimension_x + 1) == 0) {
					currentRank++;
					temp++;
				}
				map[i][j] = currentRank;
				cout << currentRank << " ";
			}
			currentRank = currentRank - temp;
			t = temp;
			cout << endl;
		}
	}

	MPI_Bcast(&map[0][0], (N - 1) * (N - 1), MPI_INT, 0, new_com);

	MPI_Datatype dt_column;
	MPI_Type_vector(end_y - start_y + 1, 1, N, MPI_DOUBLE, &dt_column);
	MPI_Type_commit(&dt_column);

	clock_t begin = clock();

	double maxDelta;
	double delta;


	do {
		int n_x = end_x - start_x + 1;
			MPI_Sendrecv(&U[end_y - 1][start_x], n_x, MPI_DOUBLE, top_neighbor, 0,
				&U[start_y - 1][start_x], n_x, MPI_DOUBLE, bottom_neighbor, 0, new_com, MPI_STATUS_IGNORE);
		
	
			MPI_Sendrecv(&U[start_y][start_x], n_x, MPI_DOUBLE, bottom_neighbor, 0,
				&U[end_y][start_x], n_x, MPI_DOUBLE, top_neighbor, 0, new_com, MPI_STATUS_IGNORE);

			MPI_Sendrecv(&U[start_y][end_x - 1], 1, dt_column, right_neighbor, 0,
				&U[start_y][start_x - 1], 1, dt_column, left_neighbor, 0, new_com, MPI_STATUS_IGNORE);

			MPI_Sendrecv(&U[start_y][start_x], 1, dt_column, left_neighbor, 0,
				&U[start_y][end_x], 1, dt_column, right_neighbor, 0, new_com, MPI_STATUS_IGNORE);

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

		MPI_Reduce(&delta, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, 0, new_com);
		MPI_Bcast(&maxDelta, 1, MPI_DOUBLE, 0, new_com);


	} while (maxDelta > EPS);

	clock_t finish = clock();

	if (rank == 0) {
		cout << "Time: " << finish - begin << endl;
	}

	// Сбор результатов в нулевой поток

	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			MPI_Bcast(&U[i][j], 1, MPI_DOUBLE, map[i][j], new_com);
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
