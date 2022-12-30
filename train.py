from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []



# charger les fichiers image à partir du jeu de données
"""
Cette ligne de code utilise la bibliothèque glob pour récupérer une liste de fichiers dans le dossier "gender_dataset_face" et ses sous-dossiers. Le paramètre "recursive=True" indique à glob de parcourir tous les sous-dossiers de manière récursive.

La fonction "glob" retourne une liste de chaînes de caractères qui correspondent aux noms de fichier trouvés dans le dossier spécifié. La liste est filtrée en utilisant une compréhension de liste, qui ne conserve que les fichiers qui ne sont pas des dossiers (ceux qui ne sont pas identifiés comme tels en utilisant "os.path.isdir").

Ensuite, la fonction "random.shuffle" de la bibliothèque "random" est utilisée pour mélanger aléatoirement l'ordre des fichiers dans la liste. Cela peut être utile si vous souhaitez par exemple créer un échantillon aléatoire de fichiers pour un traitement ultérieur.
"""
image_files = [f for f in glob.glob(r'C:\Files\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)



# convertir des images en tableaux et étiqueter les catégories
"""
Ce morceau de code parcourt tous les fichiers dans la liste "image_files" et effectue les actions suivantes pour chaque image :

La fonction "cv2.imread" est utilisée pour lire l'image à partir du chemin spécifié.

L'image est redimensionnée à l'aide de la fonction "cv2.resize", en utilisant les dimensions spécifiées dans la variable "img_dims".

L'image est convertie en un tableau NumPy en utilisant la fonction "img_to_array" de la bibliothèque "keras.preprocessing.image".

Le tableau NumPy est ajouté à la liste "data".

Le label de l'image est déterminé en analysant le chemin du fichier. Si le fichier se trouve dans le sous-dossier "woman", le label est défini comme étant 1, sinon il est défini comme étant 0. Le label est ajouté à la liste "labels" sous la forme d'une liste à une dimension (par exemple, [1] ou [0])
"""
for img in image_files:

    image = cv2.imread(img)
    
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2] # C:\Files\gender_dataset_face\woman\face_1162.jpg
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append([label]) # [[1], [0], [0], ...]



# pre-processing
"""
Ces deux lignes de code effectuent les actions suivantes :

La liste "data" est convertie en un tableau NumPy à l'aide de la fonction "np.array". Le paramètre "dtype" est défini comme "float", ce qui signifie que les valeurs de chaque pixel de l'image seront converties en nombres flottants.

Les valeurs de chaque pixel dans le tableau "data" sont divisées par 255. Cela a pour effet de normaliser les valeurs de chaque pixel entre 0 et 1, ce qui est souvent utile lors de l'entraînement de modèles d'apprentissage automatique.

La liste "labels" est également convertie en un tableau NumPy à l'aide de la fonction "np.array".

Ces étapes sont généralement effectuées pour préparer les données avant de les utiliser pour entraîner un modèle d'apprentissage automatique. La normalisation des valeurs de pixel peut aider à améliorer la performance du modèle, tandis que la conversion des données en tableaux NumPy permet de les utiliser plus facilement avec des bibliothèques de traitement d'images et d'apprentissage automatique.
"""
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)



# split dataset for training and validation
"""
Ces deux lignes de code utilisent la fonction "train_test_split" de la bibliothèque "sklearn.model_selection" pour diviser les données en deux ensembles : un ensemble d'entraînement et un ensemble de test. Le paramètre "test_size" détermine la proportion des données qui sera affectée à l'ensemble de test (dans ce cas, 20 %). Le paramètre "random_state" définit une graine aléatoire qui sera utilisée pour sélectionner aléatoirement les données dans chaque ensemble.

Ensuite, la fonction "to_categorical" de la bibliothèque "keras.utils" est utilisée pour convertir les labels d'entraînement et de test en vecteurs de catégories. Par exemple, si le label est 0, il sera converti en [1, 0], et si le label est 1, il sera converti en [0, 1]. Le paramètre "num_classes" doit être défini comme étant égal au nombre de catégories possibles (dans ce cas, 2).

Ces étapes sont généralement effectuées avant de commencer à entraîner un modèle de classification pour s'assurer que les données sont prêtes à être utilisées. La division des données en ensembles d'entraînement et de test permet de vérifier la performance du modèle sur des données qu'il n'a pas vues pendant l'entraînement, tandis que la conversion des labels en vecteurs de catégories est souvent requise pour utiliser certaines architectures de réseaux de neurone
"""
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)



# augmenting datset
"""
La ligne de code ci-dessus crée un objet "ImageDataGenerator" de la bibliothèque "keras.preprocessing.image". Cet objet peut être utilisé pour générer de nouvelles images à partir d'une base de données d'images existante, en utilisant différentes transformations d'image aléatoires.

Les transformations spécifiées dans l'objet "aug" incluent :

"rotation_range" : pour une rotation aléatoire de l'image d'un angle compris entre -25 et 25 degrés.
"width_shift_range" et "height_shift_range" : pour un déplacement aléatoire de l'image d'une largeur ou d'une hauteur comprise entre -10 % et 10 % de la taille de l'image.
"shear_range" : pour une transformation de torsion aléatoire de l'image.
"zoom_range" : pour un zoom aléatoire de l'image, allant de 80 % à 120 % de la taille de l'image.
"horizontal_flip" : pour une symétrie horizontale aléatoire de l'image.
"fill_mode" : pour définir comment les pixels manquants sont remplis lors de la transformation de l'image (ici, "nearest" signifie qu'ils sont remplis avec le pixel le plus proche).
L'objet "aug"  utilisé pour générer de nouvelles images en utilisant la méthode "flow", qui prend en entrée les données d'entraînement et génère de nouvelles images en utilisant les transformations spécifiées. Cela peut être utile pour augmenter la taille de la base de données d'entraînement et améliorer la robustesse du modèle.
"""
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")





# define model
"""
Cette fonction définit un modèle de réseau de neurones à convolution (CNN) pour la classification d'images. Le modèle est créé en utilisant la classe "Sequential" de la bibliothèque "keras.models", qui permet de définir un modèle en ajoutant des couches de manière séquentielle.

Le modèle commence par une couche de convolution à 2D (Conv2D) suivie d'une couche d'activation ReLU (Rectified Linear Unit) et d'une couche de normalisation en lots (BatchNormalization). La couche de convolution a 32 filtres de 3x3 et utilise un padding "same", ce qui signifie que les bords de l'image sont remplis de manière à ce que l'image de sortie ait la même taille que celle d'entrée. La couche de normalisation en lots permet de régulariser les activations en réduisant leur variance.

Le modèle utilise également des couches de max pooling (MaxPooling2D) pour réduire la dimensionnalité des données et des couches de dropout (Dropout) pour réduire l'overfitting. Le modèle inclut également plusieurs couches de convolution et de normalisation en lots supplémentaires avant de terminer par une couche plate (Flatten), une couche dense et une couche d'activation sigmoid.

La fonction prend en entrée les dimensions de l'image d'entrée (hauteur, largeur et profondeur) et le nombre de classes de sortie, et retourne le modèle construit.
"""
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1
    
    # L'axe qui doit être normalisé, après une couche Conv2D avec data_format="channels_first",
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model



# build model
"""
la fonction "build" définie précédemment pour construire un modèle de CNN en utilisant les dimensions spécifiées dans la variable "img_dims" et le nombre de classes de sortie (2). Le modèle construit est stocké dans la variable "model
"""
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

# compile the model
"""
Ces deux lignes de code compilent le modèle de CNN en utilisant l'algorithme d'optimisation Adam et la fonction de perte "binary_crossentropy".

L'algorithme Adam (Adaptive Moment Estimation) est un algorithme d'optimisation populaire qui utilise des estimations adaptatives des moments pour mettre à jour les poids du modèle de manière efficace. Il peut être utilisé avec des réseaux de neurones et s'adapte automatiquement aux données en utilisant des taux d'apprentissage variables.

La fonction de perte "binary_crossentropy" est une fonction de perte utilisée pour les problèmes de classification binaire. Elle mesure la distance entre la distribution prédite par le modèle et la distribution cible (les labels de sortie). Plus cette distance est faible, meilleure est la performance du modèle.

Le taux d'apprentissage (learning rate, "lr") et la décroissance du taux d'apprentissage sont également spécifiés lors de la compilation du modèle. Le taux d'apprentissage détermine la vitesse à laquelle les poids du modèle sont
"""
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
"""
Cette ligne de code utilise la méthode "fit_generator" du modèle pour entraîner le modèle en utilisant un générateur de données. Le générateur est créé en utilisant l'objet "aug" de type "ImageDataGenerator" et en utilisant la méthode "flow" pour générer de nouvelles images à partir des données d'entraînement en utilisant les transformations spécifiées dans l'objet "aug".

Le modèle est entraîné sur les données générées en utilisant le nombre d'époques spécifié (epochs) et en utilisant un lot de données de taille spécifiée (batch_size). La méthode "fit_generator" prend également en entrée les données de validation et les utilise pour évaluer la performance du modèle pendant l'entraînement.

La variable "H" contient les informations sur l'historique de l'entraînement du modèle, notamment les valeurs de perte et de précision pour chaque époque. Ces informations peuvent être utilisées pour tracer l'évolution de la performance du modèle au fil du temps et pour ajuster les
"""
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
"""
Cette ligne de code utilise la méthode "save" de l'objet modèle pour enregistrer le modèle entraîné sur le disque,Le modèle est enregistré sous le nom de fichier "gender_detection.model"
Une fois le modèle chargé, il peut être utilisé pour effectuer des prédictions sur de nouvelles données en utilisant la méthode "predict".

"""
model.save('gender_detection.model')

# plot training/validation loss/accuracy
"""
Cette ligne de code utilise la bibliothèque "matplotlib" pour tracer les valeurs de perte et de précision du modèle en fonction du nombre d'époques d'entraînement. Les valeurs de perte et de précision sont récupérées à partir de l'historique de l'entraînement du modèle, stocké dans la variable "H".

Le tracé inclut deux courbes pour chaque métrique : une pour les données d'entraînement et une pour les données de validation. Cela permet de vérifier si le modèle sur-apprend (overfitting) sur les données d'entraînement en comparant les valeurs de perte et de précision pour les données d'entraînement et de validation. Si les valeurs de perte et de précision 
des données de validation sont significativement plus élevées que celles des données d'entraînement, cela peut indiquer que le modèle sur-apprend et que sa performance sur de nouvelles données pourrait être inférieure à celle observée sur les données d'entraînement.
"""
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
"""
Cette ligne de code utilise la fonction "savefig" de la bibliothèque "matplotlib" pour enregistrer le tracé du modèle sous forme d'image au format PNG. L'image enregistrée est stockée sur le disque sous le nom de fichier "plot.png" et peut être visualisée à l'aide de n'importe quel logiciel de visualisation d'images.

Il est courant de tracer et d'enregistrer les résultats de l'entraînement d'un modèle afin de pouvoir les analyser et les comparer ultérieurement. Cela peut être particulièrement utile pour surveiller l'évolution de la performance du modèle au fil du temps et pour détecter les signes de sur-apprentissage ou de sous-apprentissage. En analysant les tracés de
la perte et de la précision, il est possible de déterminer si le modèle progresse de manière satisfaisante et de décider s'il est nécessaire de réajuster les hyperparamètres du modèle ou de changer de modèle.
"""

plt.savefig('plot.png')