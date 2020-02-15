#!/bin/bash
TIME_GLOBAL=0
TIME_SHARED=0
LINES=1024
MAXLINES=2048
INCREMENT=256

gcc argumentos.c -o ./test.o

if [ -f ./tiempos.csv ]; then
    rm ./tiempos.csv
fi

echo 'size_matrix;tiempo_global;tiempo_compartido' >> tiempos.csv

awk 'BEGIN { printf "%-7s %-20s %-20s\n", "LINES", "TIME_GLOBAL", "TIME_SHARED"}'

while [  $LINES -lt $MAXLINES ]; do

    TIME_GLOBAL="$(./test.o $LINES)"

    awk 'BEGIN { printf "%-7d %-20f %-20f\n", '$LINES', '$TIME_GLOBAL', '$TIME_SHARED'; exit; }'

    echo $LINES';'$TIME_GLOBAL';'$TIME_SHARED >> tiempos.csv

    let LINES=LINES+INCREMENT
done    

#source /home/joma/anaconda3/bin/activate tensorflow
#python grafico.py
