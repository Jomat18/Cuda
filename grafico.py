# coding=utf-8
import numpy as np 
import matplotlib.pyplot as plt 

size_matrix=[]
m_global=[]
m_shared=[]
'''
f = open('tiempos.csv', "r")
next(f)  #salto la cabecera
for linea in f: 
	if len(linea.strip())!=0:
		linea=linea.rstrip('\n')  #linea
		dato=linea.split(';')	#lista de la linea
		size_matrix.append(int(dato[0]))
		m_global.append(float(dato[1])) 
		m_shared.append(float(dato[2]))
		
	
f.close()


x = [100, 200, 300, 400, 500, 600, 700]
y1 = [0.436288, 0.900864, 1.802080, 2.902880, 5.539296, 8.498368, 13.178368] 
y2 = [0.559168, 0.665888, 1.138272, 1.811584, 2.642496, 3.699712, 5.389696] 
'''

size_matrix = [100, 200, 300, 400, 500, 600, 700]
m_global = [0.436288, 0.900864, 1.802080, 2.902880, 5.539296, 8.498368, 13.178368] 
m_shared = [0.559168, 0.665888, 1.138272, 1.811584, 2.642496, 3.699712, 5.389696] 



plt.title("Memoria Global - Memoria Compartida") 
plt.plot(size_matrix, m_global,'-b', label='GPU Global memory', marker='o', linestyle='--', fillstyle='none') 
plt.plot(size_matrix, m_shared,'-r', label='GPU Shared memory', marker='o', linestyle='--', fillstyle='none') 
plt.legend(loc='upper left')
plt.xlabel('tamaño de la matriz')
plt.ylabel('tiempo (milisegundos)')
plt.savefig('comparacion.png')
plt.show() 

