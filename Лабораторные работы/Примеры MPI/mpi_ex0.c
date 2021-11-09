/* *  ex0.c  --  пример 0       см.также  --  http://www.csa.ru/~il/mpi_tutor
 * Начало и завершение: MPI_Init, MPI_Finalize
 * Определение количества задач в приложениии и своего порядкового номера:
 * MPI_Comm_size, MPI_Comm_rank */

#include <mpi.h>
#include <stdio.h>
int main( int argc, char **argv )
{
    int size, rank, i;
      /* Инициализируем библиотеку */
    MPI_Init( &argc, &argv );
      /* Узнаем количество задач в запущенном приложении */
    MPI_Comm_size( MPI_COMM_WORLD, &size );
      /* ... и свой собственный номер: от 0 до (size-1) */
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      /* задача с номером 0 сообщает пользователю размер группы,
       * к которой прикреплена область связи,
       * к которой прикреплен описатель (коммуникатор) MPI_COMM_WORLD,
       * т.е. число процессов в приложении!!
       */
    if( rank==0 )
        printf("Total processes count = %d\n", size );
      /* Каждая задача выводит пользователю свой номер */
    printf("Hello! My rank in MPI_COMM_WORLD = %d\n", rank );
 
     /* Точка синхронизации, затем задача 0 печатает
       * аргументы командной строки. В командной строке
       * могут быть параметры, добавляемые загрузчиком MPIRUN.
       */
    MPI_Barrier( MPI_COMM_WORLD );
    if( rank == 0 )
        for( puts("Command line of process 0:"), i=0; i<argc; i++ )
            printf( "%d: \"%s\"\n", i, argv[i] );
      /* Все задачи завершают выполнение */
    MPI_Finalize();
    return 0;
}
