
 #include <pthread.h>
 #include <stdio.h>
 #include <stdlib.h>
 #define NTHREADS 10
 #define N 1000
 #define MEGEXTRA 1000000
 
 pthread_attr_t attr;
 
 void *hacer(void *thread_id)
 {
    double A[N][N];
    int i,j;
    long tid;
    size_t mi_stacksize;

    tid = (long)thread_id;
    pthread_attr_getstacksize (&attr, &mi_stacksize);
    printf("Thread %ld: tamaño de la stack = %li bytes \n", tid, mi_stacksize);
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
       A[i][j] = ((i*j)/3.452) + (N-i);
    pthread_exit(NULL);
 }
 
 int main(int argc, char *argv[])
 {
    pthread_t threads[NTHREADS];
    size_t stacksize;
    int rc;
    long t;
 
    pthread_attr_init(&attr);
    pthread_attr_getstacksize (&attr, &stacksize);
    printf("Tamaño de la stack por defecto = %li\n", stacksize);
    stacksize = sizeof(double)*N*N+MEGEXTRA;
    printf("Cantidad de stack por thread = %li\n",stacksize);
    pthread_attr_setstacksize (&attr, stacksize);
    printf("Creando pilas de tamaño = %li bytes\n",stacksize);
    for(t=0; t<NTHREADS; t++){
       rc = pthread_create(&threads[t], &attr, hacer, (void *)t);
       if (rc){
          printf("ERROR: %d\n", rc);
          exit(-1);
       }
    }
    printf("Threads creadas %ld.\n", t);
    pthread_exit(NULL);
 }

 /*
void rastrear(void)
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  printf ("Obtenido %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
     printf ("%s\n", strings[i]);

  free (strings);
}

void mi_function (void)
{
  rastrear ();
}

int main (void)
{
  mi_function ();
  return 0;
}


*/