import numpy as np
import cv2

#función para capturar camaras disponibles.
def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

print("Available camera indices:", list_cameras())


cap = cv2.VideoCapture(1)

while (cap.isOpened):
    #captura de camara dentro de video
    ret, frame = cap.read()
    #read devuelve ret y frame, ret = lectura frame = imagen(video).

    if ret:
        cv2.imshow('frame', frame)
        #salir
        if cv2.waitKey(5) & 0xFF == ('q'):
            break
    else :
        break

cap.release()
cv2.destroyAllWindows()
    

