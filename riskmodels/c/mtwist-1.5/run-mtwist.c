#include <stdio.h>
#include <stdlib.h>
#include "mtwist.h"

int main(void) {
   int i;
   mt_seed();
   for(i = 0; i < 10; ++i) {
      printf("%f\n", mt_drand());
   }
   return EXIT_SUCCESS;
}