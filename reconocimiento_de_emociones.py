# 	Nombre: reconocimiento_de_emociones.py
#	Autor: Daniel Antonio Quihuis Quihuis Hernandez
#	Fecha: Abril del 2022
#	Descripcion: Este archivo contiene el codigo para el reconocimiento de emociones

import cv2
import os
import numpy as np

def emotionImage(emotion):
	# Imagenes de referencia
	if emotion == 'Feliz': image = cv2.imread('Emociones/felicidad.jpeg')
	if emotion == 'Enfadado': image = cv2.imread('Emociones/enojo.jpeg')
	if emotion == 'Sorprendido': image = cv2.imread('Emociones/sorpresa.jpeg')
	if emotion == 'Triste': image = cv2.imread('Emociones/tristeza.jpeg')#ruta de imagen de referencia
#	if emotion == 'Neutral': image = cv2.imread('Emociones/Neutral.png') #ruta de imagen de referencia
#	if emotion == 'Asustado': image = cv2.imread('Emociones/Asustado.png') #ruta de imagen de referencia
	return image

# Modelos para entrenamiento y lectura
method = 'EigenFaces'
#method = 'FisherFaces'
#method = 'LBPH'

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('Algoritmo'+method+'.xml')

dataPath = 'C:\\Users\\KLKB\Documents\\GitHub\\S.R.F_Sistema_De_Reconocimiento_Facial\DataEmociones' #Ruta de "Data"
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:

	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cara = auxFrame[y:y+h,x:x+w]
		cara = cv2.resize(cara,(150,150),interpolation= cv2.INTER_CUBIC)
		result = emotion_recognizer.predict(cara)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# EigenFaces
		if method == 'EigenFaces':
			if result[1] < 5700:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'emociones desconocidas',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

		# FisherFaces
		if method == 'FisherFaces':
			if result[1] < 500:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

		# LBPH
		if method == 'LBPH':
			if result[1] < 60:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

	cv2.imshow('Reconocimiento de emociones',nFrame)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
