# 	Nombre: entrenamiento_emociones.py
#	Autor: Daniel Antonio Quihuis Quihuis Hernandez
#	Fecha: Abril del 2022
#	Descripcion: Este archivo contiene el codigo para el entrenamiento

import cv2
import os
import numpy as np
import time

def generarModelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenamiento para el reconocimento de caras
	print("Entrenando espere un momento ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando modelo
	emotion_recognizer.write("Algoritmo"+method+".xml")

dataPath = '' #Ruta de "Data"
emotionsList = os.listdir(dataPath)
print('Emociones guardadas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
	emotionsPath = dataPath + '/' + nameDir

	for fileName in os.listdir(emotionsPath):
		#print('Caras: ', nameDir + '/' + fileName)  # pruebas ignorar linea de codigo
		labels.append(label)
		facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))
		#image = cv2.imread(emotionsPath+'/'+fileName,0)  # pruebas ignorar linea de codigo
		#cv2.imshow('image',image)  # pruebas ignorar linea de codigo
		#cv2.waitKey(10)  # pruebas ignorar linea de codigo
	label = label + 1

generarModelo('EigenFaces',facesData,labels)
generarModelo('FisherFaces',facesData,labels)
generarModelo('LBPH',facesData,labels)
