# 	Nombre: entrenamiento_reconocimiento_facial.py
#	Autor: Daniel Antonio Quihuis Quihuis Hernandez
#	Fecha: Abril del 2022
#	Descripcion: Este archivo contiene el codigo para entrenar los 3 algoritmos

import cv2
import os
import numpy as np

dataPath = 'C:\\Users\\KLKB\\Documents\\GitHub\\S.R.F_Sistema_De_Reconocimiento_Facial\\Data' #Ruta de la "base de datos"
peopleList = os.listdir(dataPath)
print('Personas guardadas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Escaneando imagenes...')

	for fileName in os.listdir(personPath):
		print('Caras: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

#print('labels= ',labels) # pruebas ignorar linea de codigo
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0)) # pruebas ignorar linea de codigo
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1)) # pruebas ignorar linea de codigo

# Algoritmos para entrenar el reconocimiento facial
#face_recognizer = cv2.face.EigenFaceRecognizer_create() # pruebas ignorar linea de codigo
#face_recognizer = cv2.face.FisherFaceRecognizer_create()# pruebas ignorar linea de codigo
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando reconocimiento facial
print("Entrenando reconocimiento facial, espere...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('AlgoritmoEigenFaces.xml')# pruebas ignorar linea de codigo
#face_recognizer.write('AlgoritmoFisherFaces.xml')# pruebas ignorar linea de codigo
#face_recognizer.write('AlgoritmoLBPH.xml')
print("Algoritmo guardado con exito!")
