
/*
 *  ex7.c  --  пример 7       см.также  --  http://www.csa.ru/~il/mpi_tutor
 *
 *  Создание под-коммуникаторов. Неявное разбиение группы на подгруппы:
 *     MPI_Comm_split
 *
 *  Рекомендация:
 *    запустите этот пример несколько раз с разным числом ветвей N ( -np N ),
 *    например: N = 6,7,8
 */

#include <mpi.h>
#include <stdio.h>

#define  tag1  1
#define  tag2  2
#define  tag3  1   /* сознательная "ошибка": идентификатор равен 'tag1' */

#define  ELEMS(x)  (sizeof( x )/sizeof( x[0] ))

int main( int argc, char **argv )
{
    MPI_Comm subComm;
    int rank, size, subCommIndex, subRank, subSize;

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

      /* Распределяем по следующему правилу:
       * каждые три ветви - в подгруппу,
       * каждую четвертую - "в никуда"
       */
    subCommIndex = rank % 4;
    if( subCommIndex == 3 )
        subCommIndex = MPI_UNDEFINED;

      /* Желательная нумерация внутри подгрупп:
       * обратная той, что имеется в MPI_COMM_WORLD
       */
    subRank = size - rank;

      /* Вызываем во ВСЕХ ветвях-абонентах MPI_COMM_WORLD,
       * указанного первым аргументом функции
       */
    MPI_Comm_split( MPI_COMM_WORLD, subCommIndex, rank, &subComm );

      /* Каждая ветвь пишет, к чему она теперь относится
       */
    printf("My rank in MPI_COMM_WORLD = %d. ", rank);
    if( subComm == MPI_COMM_NULL )
        printf("I\'m not attached to any sub-communicator\n");
    else {
        MPI_Comm_size( subComm, &subSize );
        MPI_Comm_rank( subComm, &subRank );
        printf("My local rank = %d, local size = %d\n", subRank, subSize );
    }   

    MPI_Comm_free( &subComm );

    MPI_Finalize();
}
