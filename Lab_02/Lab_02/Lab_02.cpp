#include <mpi.h>
#include <time.h>
#include <ctime>
#include <iostream>


using namespace std;

void merge(int* a, int* b, int left, int mid, int right) {
  int h = left;
  int i = left;
  int j = mid + 1;
  while ((h <= mid) && (j <= right)) {
    if (a[h] <= a[j]) {
      b[i] = a[h];
      h++;
    } else {
      b[i] = a[j];
      j++;
    }
    i++;
  }
  if (mid < h) {
    for (int k = j; k <= right; k++) {
      b[i] = a[k];
      i++;
    }
  } else {
    for (int k = h; k <= mid; k++) {
      b[i] = a[k];
      i++;
    }
  }
  for (int k = left; k <= right; k++) {
    a[k] = b[k];
  }
}

void mergeSort(int* a, int* b, int left, int right) {
  if (left >= right) {
    return;
  }

  int mid = (left + right) / 2;

  mergeSort(a, b, left, mid);
  mergeSort(a, b, mid + 1, right);
  merge(a, b, left, mid, right);
}

int main(int argc, char** argv) {
  srand(time(NULL));
  int n;
  int main_rank;
  int processAmount;
  int* main_array{};
  int t1, t2;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &main_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &processAmount);

  if (main_rank == 0) {
    cout << "Amount of processes = " << processAmount << endl;

    cout << "Enter array size:" << endl;
    cin >> n;

    main_array = new int[n];

    cout << "Unsorted array:" << endl;
    for (int i = 0; i < n; i++) {
      main_array[i] = rand() % n;
      cout << main_array[i] << " ";
    }
    cout << endl;
  }
  t1 = clock();
  MPI_Bcast(&n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

  int size = n / processAmount;

  if (main_rank == 0) {
    size = size + (n % processAmount);
  }
  int* sub_array = new int[size];

  MPI_Scatter(main_array, size, MPI_INTEGER, sub_array, size, MPI_INTEGER, 0,
              MPI_COMM_WORLD);

  int* temp_array = new int[size];
  mergeSort(sub_array, temp_array, 0, size - 1);

  int* sorted = NULL;
  if (main_rank == 0) {
    sorted = new int[n];
  }

  MPI_Gather(sub_array, size, MPI_INTEGER, sorted, size, MPI_INTEGER, 0,
             MPI_COMM_WORLD);

  t2 = clock();
  if (main_rank == 0) {
    int* other_array = new int[n];
    mergeSort(sorted, other_array, 0, n - 1);

    cout << "Sorted array:" << endl;
    for (int i = 0; i < n; i++) {
      cout << sorted[i] << " ";
    }
    cout << endl;
    cout << "Spended time: ";
    cout << t2 - t1 << endl;
    delete[] other_array;
  }

  delete[] sorted;
  delete[] main_array;
  delete[] sub_array;
  delete[] temp_array;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
