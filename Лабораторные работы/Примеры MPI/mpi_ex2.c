/*
 *  ex2.c  --  пример 2       см.также  --  http://www.csa.ru/~il/mpi_tutor
 *
 *  Прием сообщений неизвестной длины:
 *      MPI_Probe
 *
 *  Прием сообщений от разных отправителей с разными идентификаторами
 *  ( с содержимым разных типов ) в произвольном порядке:
 *      MPI_ANY_SOURCE, MPI_ANY_TAG     ( джокеры )
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

  /* Идентификаторы сообщений */
#define tagFloatData  1
#define tagLongData   2

  /* Длина передаваемых сообщений может быть
   * случайной от 1 до maxMessageElems
   */
#define maxMessageElems 100

int main( int argc, char **argv )
{
    int    size, rank, count, i, n, ok;
    float *floatPtr;
    int   *longPtr;
    char  *typeName;
    MPI_Status status;

      /* Инициализация и сообщение об ошибке
       * целиком перенесены из первого примера
       */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

      /* пользователь должен запустить ровно ТРИ задачи, иначе ошибка */
    if( size != 3 ) {
        if( rank==0 )
            printf("Error: 3 processes required instead of %d\n", size );
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Abort( MPI_COMM_WORLD, MPI_ERR_OTHER );
        return -1;
    }

      /* Каждая задача инициализирует генератор случайных чисел */
    srand( ( rank + 1 ) * (unsigned )time(0) );

    switch( rank ) {

        case 0:
              /* создаем сообщение случайной длины */
            count = 1 + rand() % maxMessageElems;
            floatPtr = malloc( count * sizeof(float) );
            for( i=0; i<count; i++ )
                floatPtr[i] = (float)i;

              /* Посылаем сообщение в задачу 2 */
            MPI_Send( floatPtr, count, MPI_FLOAT,
                2, tagFloatData, MPI_COMM_WORLD );
            printf("%d. Send %d float items to process 2\n",
                rank, count );
            break;

        case 1:
              /* создаем сообщение случайной длины */
            count = 1 + rand() % maxMessageElems;
            longPtr = malloc( count * sizeof(long) );
            for( i=0; i<count; i++ )
                longPtr[i] = i;

              /* Посылаем сообщение в задачу 2 */
            MPI_Send( longPtr, count, MPI_LONG,
                2, tagLongData, MPI_COMM_WORLD );
            printf("%d. Send %d long items to process 2\n",
                rank, count );
            break;

        case 2:
            /* задача 2 принимает сообщения неизвестной длины,
             * используя MPI_Probe
             */
            for( n=0; n<2; n++ )  /* Всего ожидаются два сообщения */
            {
                MPI_Probe(
                    MPI_ANY_SOURCE,  /* Джокер: ждем от любой задачи */
                    MPI_ANY_TAG,     /* Джокер: ждем с любым идентификатором */
                    MPI_COMM_WORLD,
                    &status );
                /* MPI_Probe вернет управление, когда сообщение будет
                 * уже на приемной стороне в служебном буфере
                 */

                /* Проверяем идентификатор и размер пришедшего сообщения
                 */
                if( status.MPI_TAG == tagFloatData )
                {
                    MPI_Get_count( &status, MPI_FLOAT, &count );

                    /* Принятое будет размещено в динамической памяти:
                     * заказываем в ней буфер соответствующей длины
                     */
                    floatPtr = malloc( count * sizeof(float) );

                    MPI_Recv( floatPtr, count, MPI_FLOAT,
                        MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                        &status );
                    /* MPI_Recv просто скопирует уже принятые данные
                     * из системного буфера в пользовательский
                     */

                    /* Проверяем принятое */
                    for( ok=1, i=0; i<count; i++ )
                        if( floatPtr[i] != (float)i )
                            ok = 0;
                    typeName = "float";
                }
                else if( status.MPI_TAG == tagLongData )
                {
                    /* Действия, аналогичные вышеописанным
                     */
                    MPI_Get_count( &status, MPI_LONG, &count );
                    longPtr = malloc( count * sizeof(long) );

                    MPI_Recv( longPtr, count, MPI_LONG,
                        MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                        &status );

                    for( ok=1, i=0; i<count; i++ )
                        if( longPtr[i] != i )
                            ok = 0;
                    typeName = "long";
                }

                 /* Докладываем о завершении приема */
                printf( "%d. %d %s items are received from %d : %s\n", rank,
                    count, typeName, status.MPI_SOURCE, ok ? "OK" : "FAILED" );

            } /* for(n) */
            break;

    } /* switch(rank) */

     /* Завершение работы */
    MPI_Finalize();
    return 0;
}
