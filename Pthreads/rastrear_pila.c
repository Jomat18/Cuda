#include <execinfo.h>
#include <unistd.h> 
#include <stdio.h>
#include <stdlib.h>

#define MAX 200

//gcc rastrear_pila.c -rdynamic

void print_stack_rastrear()
{
  void *buffer[MAX];
  int tam;
 
  tam = backtrace(buffer, MAX);
  fprintf(stderr, "--- (Profundidad %d) ---\n", tam);
  backtrace_symbols_fd(buffer, tam, STDERR_FILENO);
}
 
void interior(int k)
{
  print_stack_rastrear();
}
 
void medio(int x, int y)
{
  interior(x*y);
}
 
void exterior(int a, int b, int c)
{
  medio(a+b, b+c);
}
 
int main()
{
  exterior(2,3,5);
  return EXIT_SUCCESS;
}
