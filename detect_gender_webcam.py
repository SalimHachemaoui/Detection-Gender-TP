from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
                    
# load model
"""
Cette ligne de code utilise la fonction "load_model" de la bibliothèque "keras.models" pour charger un modèle pré-entraîné à partir du disque. Le modèle est chargé à partir du fichier "gender_detection.model" et est stocké dans la variable "model".

Une fois le modèle chargé, il peut être utilisé pour effectuer des prédictions sur de nouvelles données en utilisant la méthod
"""
model = load_model('gender_detection.model')

# open webcam

"""
Cette ligne de code utilise la fonction "VideoCapture" de la bibliothèque "cv2" (OpenCV) pour ouvrir la webcam par défaut de l'ordinateur. La webcam est représentée par un objet "VideoCapture" qui peut être utilisé pour capturer des images en temps réel à l'aide de la méthode "read".

La liste "classes" contient les différentes classes de sortie possibles du modèle de détection de genre. Dans ce cas, il y a deux classes possibles : "man" et "woman". Ces classes seront utilisées pour convertir les prédictions du modèle en labels lisibles par l'homme.
"""
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']

# loop through frames
"""
Ce code utilise la webcam et un modèle de détection de genre pour effectuer des prédictions de genre en temps réel sur des visages détectés dans les images capturées par la webcam.

La boucle "while" exécute le code qui se trouve à l'intérieur de la boucle jusqu'à ce que la webcam soit fermée ou que la boucle soit interrompue. À chaque itération de la boucle, une image est capturée à partir de la webcam à l'aide de la méthode "read" de l'objet "VideoCapture" et est envoyée à la fonction de détection de visage "detect_face".

La fonction "detect_face" utilise une technique de détection de visage pour localiser les visages dans l'image et renvoie un tableau de coordonnées de visage et un tableau de scores de confiance correspondants. La boucle "for" parcourt chaque visage détecté dans l'image et dessine un rectangle autour du visage à l'aide de la fonction "rectangle" de OpenCV.

Le visage est ensuite recadré à l'aide de la fonction "np.copy" et est prétraité pour être utilisé avec le modèle de détection de genre. Le modèle de détection de genre est utilisé pour effectuer une prédiction de genre sur le visage recadré à l'aide de la méthode "predict" et la prédiction est affichée sur l'image à l'aide de la fonction "putText" de OpenCV.

Enfin, l'image est affichée à l'écran à l'aide de la fonction "imshow" de OpenCV et la boucle est interrompue lorsque l'utilisateur appuie.
"""

while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
"""
Ces lignes de code libèrent les ressources de la webcam et ferment toutes les fenêtres ouvertes par OpenCV une fois que la boucle "while" est terminée.

La méthode "release" de l'objet "VideoCapture" libère les ressources de la webcam et permet à l'ordinateur de les utiliser pour d'autres tâches. La fonction "destroyAllWindows" de OpenCV ferme toutes les fenêtres ouvertes par la bibliothèque.

Il est important de libérer les ressources de la webcam et de fermer toutes les fenêtres lorsque vous avez terminé de travailler avec elles afin de libérer de la mémoire et de ne pas ralentir votre ordinateur.
"""
webcam.release()
cv2.destroyAllWindows()