import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)  # Mostrar todo el array sin truncar

img = cv.imread('gato.jpg', cv.IMREAD_GRAYSCALE)

def reduc_amplia():
    #Factores de reducción y ampliación
    reduc = [2, 3, 4]
    amplia = [10, 12, 16]

    for i in reduc:
        
        M = np.array([[1/i, 0, 0], [0, 1/i, 0]], dtype=np.float32) # Creación matriz de escalamiento
        

        # Calcular las dimensiones de la imagen reducida
        reduced_dimensions = (img.shape[1] // i, img.shape[0] // i)
        
        # Aplicar la transformación afín
        reduced_img = cv.warpAffine(img, M, reduced_dimensions, flags=cv.INTER_LINEAR)
        
        # Mostrar la imagen reducida
        cv.imshow(f'Imagen reducida = {reduced_dimensions}', reduced_img)
        cv.waitKey(0)
        cv.destroyWindow(f'Imagen reducida = {reduced_dimensions}')


    for i in amplia:
        
        M = np.array([[i, 0, 0], [0, i, 0]], dtype=np.float32) # Creación matriz de escalamiento
        
        ampliada_dimensions = (img.shape[1] * i, img.shape[0] * i)

        ampliada_img = cv.warpAffine(img, M, ampliada_dimensions, flags=cv.INTER_LINEAR)

        cv.imshow(f'Imagen ampliada = {ampliada_dimensions}', ampliada_img)
        cv.waitKey(0)
        cv.destroyWindow(f'Imagen ampliada = {ampliada_dimensions}')

def rotación():
    grados = [47, 75, 90, 135]

    for i in grados:

        radianes = np.radians(i)

        M = np.array([[np.cos(radianes), -np.sin(radianes), 0], [np.sin(radianes), np.cos(radianes), 0]], dtype = np.float32) # Creación matriz de escalamiento

        #Aplicar transformación
        rotated_img = cv.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)

        # Mostrar la imagen reducida
        cv.imshow(f'Imagen rotada grados = {i}', rotated_img)
        cv.waitKey(0)
        cv.destroyWindow(f'Imagen rotada grados = {i}')

def histograma(_img, bins = 256):

    hist = np.zeros(bins, dtype = int) # Crear un arreglo de 256 elementos
    data = _img.flatten() # Convertir la imagen a un arreglo de 1D

    for pixel in data: 
        hist[pixel] += 1

    return hist

def mostrar_imagen_y_histograma(_img, name = 'Imagen', bins = 256):
    hist = histograma(_img, bins)

    # Crear una figura con dos subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mostrar la imagen en el primer subplot
    axs[0].imshow(_img, cmap='gray')
    axs[0].set_title('Imagen')
    axs[0].axis('off')

    # Mostrar el histograma en el segundo subplot
    axs[1].bar(range(bins), hist, width=1)
    axs[1].set_title('Histograma')
    axs[1].set_xlabel('Bins')
    axs[1].set_ylabel('Frecuencia')

    fig.canvas.manager.set_window_title(name)
    plt.show()

def img_negativo():
    img_negativo = np.uint8(255 - img)
    mostrar_imagen_y_histograma(img_negativo, 'Imagen Negativa')

def img_aumentobrillo():

    img_aclarada = np.uint8(np.sqrt(255 * np.double(img)))
    mostrar_imagen_y_histograma(img_aclarada, "Imagen Aclarada")

def img_disminucionbrillo():
    img_oscura = np.uint8((np.double(img)) ** 2 / 255)
    mostrar_imagen_y_histograma(img_oscura, "Imagen Oscura")

def img_correciongamma(valor_gamma):
    gamma = valor_gamma
    img_gamma = np.uint8(255 * (img / 255) ** (1/gamma))
    mostrar_imagen_y_histograma(img_gamma, f"Imagen Corregida Gamma = {gamma}")

def normalizar_imagen(rango):
    minimo = np.min(rango)
    maximo = np.max(rango)

    img_normalizada = np.intc((maximo - minimo) * ((img - np.min(img)) / (np.max(img) - np.min(img))) + minimo)
    mostrar_imagen_y_histograma(img_normalizada, f"Imagen Normalizada Rango = {rango}", np.max(img_normalizada) + 1)


#def contraste_logaritmico():




mostrar_imagen_y_histograma(img)
reduc_amplia()
rotación()
img_negativo()
img_aumentobrillo()
img_disminucionbrillo()
img_correciongamma(0.1)
normalizar_imagen([0, 100])
normalizar_imagen([-1, 1])
normalizar_imagen([0, 1024])