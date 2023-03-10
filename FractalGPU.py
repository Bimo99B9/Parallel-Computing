import sys

from PIL import Image
import ctypes
import numpy as np 
from numpy import linalg as LA
from numpy.ctypeslib import ndpointer
from time import time



#########################################################################
# Preparar gestión librería externa de Profesor	 			#
#########################################################################
libProf = ctypes.cdll.LoadLibrary('./mandelProfGPU.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None"
# .argtypes se refiere al tipo de los argumentos de la funcion
# Hay dos funciones en la librería compartida: mandel(...) y mandelPar(...)
mandelProf = libProf.mandelGPU

mandelProf.restype  = None
mandelProf.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "double promedio(int, int, double *)
mediaProf = libProf.promedioGPU

mediaProf.restype  = ctypes.c_double
mediaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "void binariza(int, int, double *)
binarizaProf = libProf.binarizaGPU

binarizaProf.restype  = None
binarizaProf.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double,ctypes.c_int]

#########################################################################
# Preparar gestión librería externa de Alumnx mandelGPU.so		#
#########################################################################
libAl = ctypes.cdll.LoadLibrary('./mandelGPU.so')
# Preparando para el uso de "void mandel(double, double, double, double, int, int, int, double *)
# .restype  se refiere al tipo de lo que retorna la funcion. Si es void, valor "None"
# .argtypes se refiere al tipo de los argumentos de la funcion
# Hay dos funciones en la librería compartida: mandel(...) y mandelPar(...)
mandelAl = libAl.mandelGPU

mandelAl.restype  = None
mandelAl.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "double promedio(int, int, double *)
mediaAl = libAl.promedioGPU

mediaAl.restype  = ctypes.c_double
mediaAl.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_int]

# Preparando para el uso de "void binariza(int, int, double *)
binarizaAl = libAl.binarizaGPU

binarizaAl.restype  = None
binarizaAl.argtypes = [ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_double,ctypes.c_int]

#########################################################################
# 	Función para guardar la imagen a archivo			#
#########################################################################
def grabar(vect, xres, yres, nom):
    A2D=vect.astype(np.ubyte).reshape(yres,xres) #(filas,columnas)
    im=Image.fromarray(A2D)
    im.save(nom)


#########################################################################
# 			MAIN						#
#########################################################################   
if __name__ == "__main__":
    # Proceado de los agumentos			#
    if len(sys.argv) != 8:
        print('FractalGPU.py <xmin> <xmax> <ymin> <yres> <maxiter> <ThpBlk> <outputfile>')
        print("Ejemplo: -0.7489 -0.74925 0.1007 1024 1000 32 out.bmp")
        sys.exit(2)
    outputfile = sys.argv[7]
    outPy="py"+outputfile
    xmin=float(sys.argv[1])
    xmax=float(sys.argv[2])
    ymin=float(sys.argv[3])
    yres=int(sys.argv[4])
    maxiter=int(sys.argv[5])
    ThpBlk=int(sys.argv[6])	# ThpBlk es nº de hilos en cada dimensión.
    
    # Cálculo de otras variables necesarias					#
    xres = yres
    ymax = ymin+(xmax-xmin)
    reps = 1
    reps_prom = 1
    reps_bin = 1
    
    # Reserva de memoria de las imágenes en 1D					#
    fractalAl = np.zeros(yres*xres).astype(np.double)
    fractalC = np.zeros(yres*xres).astype(np.double)

    
    print(f'\nEjecutando {yres}x{xres}')
    
    # Llamadas a las funciones de cálculo del fractal Prof (NO MODIFICAR)	#
    sC = time()
    mandelProf(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalC, ThpBlk)
    sC = time()- sC
    print(f"mandelProfGPU ha tardado {sC:1.5E} segundos")
    
    # Llamadas a las funciones de cálculo del fractal	Alum, guardar en fractalAl#
    sC = time()
    for i in range(reps): 
        mandelAl(xmin, ymin, xmax, ymax, maxiter, xres, yres, fractalAl, ThpBlk)
    sC = time()- sC
    tiempo = sC/reps
    print(f"mandelAlGPU ha tardado ={tiempo:1.5E} segundos")
    
    
    # Comprobación de los errores			(NO MODIFICAR)		#
    print('El error es '+ str(LA.norm(fractalAl-fractalC)))
    print("")


    # Llamadas a las funciones de cálculo de la promedio Prof (NO MODIFICAR)	#
    sP = time()
    for i in range(reps_prom):
        mediaPr = mediaProf(xres, yres, fractalC, ThpBlk)
    sP = time()- sP
    tiempo = sP/reps_prom
    print(f"mediaProfGPU={mediaPr} y	ha tardado {tiempo:1.5E} segundos")

    
    # Llamadas a las funciones de cálculo del promedio Alum			#
    sP = time()
    for i in range(reps_prom):
        mediaAll = mediaAl(xres, yres, fractalAl, ThpBlk)
    sP = time()- sP
    tiempo = sP/reps_prom
    print(f"mediaAlGPU={mediaAll} y	ha tardado ={tiempo:1.5E} segundos")
    
    # Comprobación de los errores						#
    print('El error es '+ str(LA.norm(mediaPr - mediaAll)))
    print("")

    # Llamadas a las funciones de cálculo de la binarizado Prof (NO MODIFICAR)	#
    sB = time()
    binarizaProf(xres, yres, fractalC, mediaPr, ThpBlk)
    sB = time()- sB
    print(f"binarizaProfGPU	ha tardado {sB:1.5E} segundos")

    # grabar(fractalAl,xres,yres,"AlNoBin"+outputfile)
    
    # Llamadas a las funciones de cálculo del binarizado Alum			#
    sB = time()
    binarizaAl(xres, yres, fractalAl, mediaAll, ThpBlk)
    sB = time()- sB
    print(f"binarizaAlGPU ha tardado ={sB:1.5E} segundos")
    
    # Comprobación de los errores						#
    print('El error es '+ str(LA.norm(fractalAl-fractalC)))
    print("")

    sB = time()
    for i in range(reps_bin):
        binarizaAl(xres, yres, fractalAl, mediaAll, ThpBlk)
    sB = time()- sB
    tiempo = sB/reps_bin
    print(f"binarizaAlGPU (Repeticiones) ha tardado ={tiempo:1.5E} segundos")


    # Test
    # fractalTest= np.zeros(yres*xres).astype(np.double)
    # for i in range(len(fractalC)):
    #     if(fractalAl[i] != fractalC[i]):
    #         fractalTest[i] = 255
    #     else:
    #         fractalTest[i] = 0
    

    # Grabar a archivos (nunca usar si yres>2048)				#
    # grabar(fractalC,xres,yres,"Prof"+outputfile)
    # grabar(fractalAl,xres,yres,"Al"+outputfile)

#    grabar(fractalTest,xres,yres,"Test"+outputfile)

    

    
   
