# 	Nombre: capturando_caras_emociones.py
#	Autor: Daniel Antonio Quihuis Quihuis Hernandez
#	Fecha: Abril del 2022
#	Descripcion: Este archivo contiene el codigo que captura los rostros para el reconocimiento de emociones

import cv2
import os
import imutils

#emotionName = 'Feliz'
#emotionName = 'Enfadado'
#motionName = 'Sorprendido'
#emotionName = 'Triste'
emotionName = 'Neutral'
#emotionName = 'Asustado'

dataPath = '' #Ruta de "Data"
emotionsPath = dataPath + '/' + emotionName

if not os.path.exists(emotionsPath):
	print('Directorio creado con exito!: ',emotionsPath)
	os.makedirs(emotionsPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:

	ret, frame = cap.read()
	if ret == False: break
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(emotionsPath + '/cara_{}.jpg'.format(count),rostro)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	if k == 27 or count >= 275:
		break

cap.release()
cv2.destroyAllWindows()
