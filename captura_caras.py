# 	Nombre: captura_caras.py
#	Autor: Daniel Antonio Quihuis Quihuis Hernandez
#	Fecha: Abril del 2022
#	Descripcion: Este archivo contiene el codigo que nos permite capturar caras

import cv2
import os
import imutils

personName = 'Daniel' #etiqueta o nombre de la persona
dataPath = '/Users/antonioquihuis/Documents/GitHub/S.R.F_Sistema_De_Reconocimiento_Facial/Data' #Ruta de la carpeta Data
personPath = dataPath + '/' + personName

if not os.path.exists(personPath): #Se revisa que el directorio este creado en el sistema operativo, si no lo esta lo va a creear
	print('Directorio creado con exito: ',personPath)
	os.makedirs(personPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('videoPruebaSRF.mp4') # pruebas ignorar linea de codigo

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
		cara = auxFrame[y:y+h,x:x+w]
		cara = cv2.resize(cara,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(personPath + '/cara{}.jpg'.format(count),cara)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	if k == 27 or count >= 300:
		break

cap.release()
cv2.destroyAllWindows()
