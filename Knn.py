import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#D
# Con la biblioteca de pandas podemos hacer la manipulacion y analisis de los datos ya que con este podremos leer
# el archivo csv playwheather, con train test split y el uso de la bibilioteca sklearn podremos dividir los datos
# en conjunto de entrenamiento y pruebas


#  Esta funcion se encarga de leer los datos del archivo csv playweather usando la biblioteca de pandas y luego los
#  retorna como un data frame
def cargar_datos(ruta_archivo):
    print("Cargando datos...")
    datos = pd.read_csv(ruta_archivo)
    print("Datos cargados exitosamente.")
    print(datos)
    return datos
#C
#Esta funcion se encarga de codificar las colunmnas cateogricas  en valores numericos, separa las caracteristicas de X
#y Y luego normaliza las caracteristicas
def preparar_datos(datos):
    print("Preparando datos...")
    le = LabelEncoder()
    for columna in ['Outlook', 'Windy', 'Play']:
        datos[columna] = le.fit_transform(datos[columna])

    X = datos.drop('Play', axis=1)
    y = datos['Play']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Datos preparados.")
    return X, y

# Esta funcion se encarga de solicitra al usuario el porcentaje de datos que desea usar para entrenamiento y se encarga de validar
# que el valor solicitado este dentro del rango solicitado
def obtener_porcentajes():
    while True:
        try:
                ##Obtener el porcentaje de entrenamiento
            entrenamiento_porcentaje = float(input("Introduzca el porcentaje para entrenamiento (30 - 90): ")) / 100
            if not (0.3 <= entrenamiento_porcentaje <= 0.9):
                raise ValueError("El porcentaje esta fuera de rango. Debe ser entre 30 y 90.")
            return entrenamiento_porcentaje
        except ValueError as e:
            print(f"Entrada inválida: {e}")
#S
#Esta función divide los datos en conjuntos de entrenamiento y prueba,
# En esta funcion se entrena un clasificador KNN y predice las etiquetas del dataset de prueba y muestra la precisión.
def entrenar_y_evaluar_knn(X, y, entrenamiento_porcentaje):
    prueba_porcentaje = 1 - entrenamiento_porcentaje

    print(f"Dividiendo datos en {int(entrenamiento_porcentaje * 100)}% entrenamiento y {int(prueba_porcentaje * 100)}% prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=prueba_porcentaje, random_state=42)

    modelo = KNeighborsClassifier(n_neighbors=3)
    modelo.fit(X_train, y_train)

    print("Realizando predicciones...")
    predicciones = modelo.predict(X_test)

    precision = accuracy_score(y_test, predicciones)
    print(f'Precisión del modelo: {precision * 100:.2f}%')


# Finalmente, calculamos y mostramos la precision promedio del modelo.
def main():
    ruta_archivo = "C:/Users/Cristian Lopez/Desktop/knn/pythonProject1/playweather.csv"
    while True:
        datos = cargar_datos(ruta_archivo)
        X, y = preparar_datos(datos)
        entrenamiento_porcentaje = obtener_porcentajes()
        entrenar_y_evaluar_knn(X, y, entrenamiento_porcentaje)

        repetir = input("¿Desea repetir el programa? (1 para repetir, 2 para salir): ")
        if repetir == '2':
            print("Programa finalizado.")
            break
        elif repetir == '1':
            print("Reiniciando el programa.")
        else:
            print("Entrada no válida. Saliendo del programa.")
            break

#fin del codigo
if __name__ == "__main__":
    main()
