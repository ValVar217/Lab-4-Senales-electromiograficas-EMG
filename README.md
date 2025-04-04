<h1 align="center"> Lab 4 - Señales Electromiográficas (EMG) </h1>    
<h2 align="center"> 💥 Fatiga Muscular 💪 </h2>   

# INTRODUCCIÓN    
Mediante el desarrollo del presente informe, se muestra la elavoracion de la práctica de laboratorio enfocada en el procesamiento de señales electromiográficas (EMG) con el objetivo de poder detectar la fatiga muscular mediante el análisis espectral. La señal EMG, que representa la actividad eléctrica de los músculos, pues esta durante el laboratorio fue adquirida utilizando electrodos de superficie conectados a un sistema de adquisición de datos (DAQ), durante una contracción muscular sostenida hasta la aparición de fatiga. Posteriormente, la señal que fue previamente capturada se proceso aplicando filtros pasa altas y pasa bajas para eliminar componentes de ruido, y segmentada mediante técnicas de aventanamiento, utilizando específicamente las ventanas de Hanning y Hamming. Añadiendo que a cada segmento se le aplicó la Transformada Rápida de Fourier (FFT) para lograr obtener el espectro de frecuencias, lo que permitió calcular estadísticos característicos como la frecuencia mediana, empleada como indicador clave del nivel de fatiga muscular. El propósito de este laboratorio es que se desarrollen competencias para el optimo análisis de señales EMG desde la captura de estas hasta su interpretación espectral, todo esto para evaluar la respuesta muscular en tiempo real.

<h1 align="center"> 📄 GUIA DE USUARIO 📄 </h1>    

## ✔️ANALISIS Y RESULTADOS    
## Captura de la Señal EMG:    
Para la adquisición de la señal electromiográfica (EMG), se diseñó e implementó una interfaz gráfica en Python utilizando la biblioteca PyQt6 en combinación con PyDAQmx para la comunicación con la tarjeta DAQ. Esta interfaz nos permite la visualización en tiempo real de la señal EMG y su almacenamiento para su porterior análisis. A continuación, mostraremos cada componente que utilizamos para el desarrollo del codigo y la captura de ya mencionada señal:  

![WhatsApp Image 2025-04-04 at 1 11 24 PM](https://github.com/user-attachments/assets/992049ed-3ebd-4fae-8bcd-61a36a3bd5b8)   
  |*Figura 1: Medición de la Fatiga muscular en tiempo real.*| 
___________________________________  
El código comienza con la importación de librerías que utilizaremos para ciertos parametros en especifico como lo siguiente:
Se incluyen **sys**, **numpy** y **time** para el manejo del sistema, operaciones numéricas y la temporización. **ctypes** para gestionar tipos de datos C necesarios en la interfaz **DAQ**, y **csv** para el almacenamiento de los datos adquiridos. Desde **PyQt6**, lo que queremos es la importación de los componentes gráficos para poder construir la interfaz, y desde **pyqtgraph**, un módulo para las gráficas en tiempo real. Por ultimo tenemos la libreria **PyDAQmx**, pues esta se emplea para interactuar con el hardware de la adquisición de datos, y **scipy.signal** proporciona herramientas para el diseño y aplicación de los filtros digitales.

```python
import sys
import numpy as np
import ctypes
import csv
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtCore import QTimer
from pyqtgraph import PlotWidget, mkPen
from PyDAQmx import Task, DAQmx_Val_Diff, DAQmx_Val_ContSamps, DAQmx_Val_Rising, DAQmx_Val_GroupByChannel, DAQmx_Val_Volts
from scipy.signal import butter, lfilter  
 ```
  
Luego, se definieron los parámetros clave del sistema. Pues, en estos podemos ver **la frecuencia de muestreo** que se establece en 1000 Hz, la cosideramos adecuada para registrar señales EMG. Tambie, se configura un **filtro pasa-altas** de 10 Hz ( con el fin de eliminar componentes de baja frecuencia como el del modulo ECG) y un **filtro pasa-bajas** de 500 Hz, que limita el espectro superior a la banda del EMG. Se determino el tamaño del búfer de visualización (BUFFER_SIZE) y el tamaño total de almacenamiento en memoria (DATA_BUFFER_SIZE), que es equivalente a 10 segundos de datos. También se definen los canales analógicos como lo son el rango del eje Y del gráfico, la ruta del archivo de salida (emg_data.csv) y el intervalo de actualización de la gráfica.    

```python  
# Parámetros de adquisición y filtrado
FS = 1000  # Frecuencia de muestreo (Hz)
HP_CUTOFF = 10  # Se subió el filtro pasa-altas a 20 Hz para eliminar ECG
LP_CUTOFF = 500  # Filtro pasa-bajas (Hz)
BUFFER_SIZE = 100  # Se redujo el buffer para actualizar más rápido
DATA_BUFFER_SIZE = FS * 10  # Almacenar 10 segundos de datos
CHANNELS = ["Dev1/ai0", "Dev2/ai0"]  # Posibles canales de la DAQ
FIXED_YLIM = (-4.0, 4.0)  # Escala fija para el eje Y
DATA_FILE = "emg_data.csv"  # Archivo de almacenamiento
UPDATE_INTERVAL = 20  # Intervalo de actualización en ms (más rápido)
 ```

Posteriormente, se diseña un filtro digital de tipo **Butterworth de cuarto orden** con paso de banda entre 10 y 500 Hz. Este se realizo con el fin de eliminar componentes de frecuencia fuera del rango útil de la EMG, como el del modulo ECG o el ruido eléctrico como tal. La función **filter_signal(data)¨¨ se utiliza luego para aplicar este filtro a los datos leídos desde la DAQ. Este es un paso esencial para que la señal graficada esté limpia y nos sea confiable para su respectivo análisis..

```python
# Diseño del filtro Butterworth
b, a = butter(4, [HP_CUTOFF / (0.5 * FS), min(LP_CUTOFF / (0.5 * FS), 0.99)], btype='band', analog=False)
 ```
Ahora, la siguiente clase maneja la configuración del dispositivo DAQ (**EMGAcquisition**). Intenta conectarse a uno de los canales definidos en CHANNELS, configurando cada uno como canal analógico diferencial para medir voltajes entre -5V y +5V. Se ajusta el reloj de muestreo para que tome datos continuamente con la frecuencia especificada. Si ningún canal es válido, lanza un error crítico. Una vez configurado, se inicia la tarea (StartTask()) para empezar a capturar datos de la señal EMG.

  ```python  
def filter_signal(data):
    return lfilter(b, a, data)

class EMGAcquisition(Task):
    def __init__(self):
        super().__init__()
        self.device_found = False
        for ch in CHANNELS:
            try:
                self.CreateAIVoltageChan(ch, "", DAQmx_Val_Diff, -5.0, 5.0, DAQmx_Val_Volts, None)
                self.CfgSampClkTiming("", FS, DAQmx_Val_Rising, DAQmx_Val_ContSamps, BUFFER_SIZE)
                self.device_found = True
                self.channel = ch
                break
            except:
                continue
        if not self.device_found:
            raise RuntimeError("No se encontró un dispositivo DAQ disponible.")
        self.StartTask()  
```
Y el **read_data()**, se encarga de leer los datos recientes de la señal EMG desde la DAQ. Tambien, se crea un buffer (data) para almacenar temporalmente las muestras. Si no se logran leer suficientes datos, se devuelven ceros para evitar algun tipo de errores. Teniendo en cuenta condiciones como la de que si la lectura es exitosa, se aplica el filtro digital a los datos, pues esta función entrega los datos ya filtrados que posteriormente serán graficados.

  ```python    
    def read_data(self):
        data = np.zeros(BUFFER_SIZE, dtype=np.float64)
        read = ctypes.c_int32(0)
        self.ReadAnalogF64(BUFFER_SIZE, 10.0, DAQmx_Val_GroupByChannel, data, BUFFER_SIZE, ctypes.byref(read), None)

        # Si no se leyeron suficientes muestras, devolver ceros para evitar ruido
        if read.value < BUFFER_SIZE:
            return np.zeros(BUFFER_SIZE)

        return filter_signal(data)  
 ```
**Interfaz Grafica: (EMGPlot)**  
Esta clase lo que hace es que define la ventana gráfica de la aplicación, pues en esta parte a continuacion, se configura un **PlotWidget** de pyqtgraph con fondo blanco, ejes etiquetados y una rejilla para facilitar la lectura de los datos mediante la captura de la señal. Tenemos tambie, la serie de datos (self.series) que es en donde se mostrará la señal EMG en tiempo real, realizando igual la inicialización de un buffer (self.data) que tiene como proposito almacenar los últimos 10 segundos de señal, junto con el temporizador (QTimer) llama periódicamente a update_plot() para actualizar la gráfica.   
Aquí es donde realmente podemos evidenciar la lectura como grafica la señal EMG:  

```python
class EMGPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG en Tiempo Real")
        self.setGeometry(100, 100, 800, 500)

        self.graphWidget = PlotWidget()
        self.graphWidget.setBackground("w")
        self.graphWidget.setTitle("Señal EMG")
        self.graphWidget.setLabel("left", "Voltaje (V)")
        self.graphWidget.setLabel("bottom", "Tiempo (ms)")
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setYRange(*FIXED_YLIM)

        self.series = self.graphWidget.plot([], [], pen=mkPen(color='r', width=2))
        self.data = np.zeros(DATA_BUFFER_SIZE)
        self.timestamps = np.linspace(-10, 0, DATA_BUFFER_SIZE)
        self.start_time = time.time()

        layout = QVBoxLayout()
        layout.addWidget(self.graphWidget)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        try:
            self.task = EMGAcquisition()
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", str(e))
            sys.exit(1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UPDATE_INTERVAL)
 ```
![WhatsApp Image 2025-03-29 at 5 07 54 PM](https://github.com/user-attachments/assets/76c51a59-d203-4ed1-8da8-b6249cf1d1ed)  
___________________________________    
![WhatsApp Image 2025-03-29 at 5 08 01 PM (1)](https://github.com/user-attachments/assets/982fdbf0-611f-4272-a409-a15383a80601)  
___________________________________  
![WhatsApp Image 2025-03-29 at 5 08 10 PM (1)](https://github.com/user-attachments/assets/13699e88-329d-4ef4-97a4-e3ebca0f2e58)    
___________________________________  


A continuación, tenemos el metodo que es llamado periódicamente por el temporizador y se encarga de actualizar la gráfica con los nuevos datos leídos desde la DAQ. Tambien se calculan los nuevos tiempos, actualiza el buffer circular y redibuja la señal. Además, llama a save_data() para guardar los datos continuamente. Aquí es donde ocurre la visualización dinámica en tiempo real de la EMG, actualizándose cada 20 ms.

```python
    def update_plot(self):
        new_data = self.task.read_data()
        current_time = time.time() - self.start_time
        new_timestamps = np.linspace(current_time - len(new_data) / FS, current_time, len(new_data))

        self.data = np.roll(self.data, -len(new_data))
        self.data[-len(new_data):] = new_data
        self.timestamps = np.roll(self.timestamps, -len(new_data))
        self.timestamps[-len(new_data):] = new_timestamps

        self.series.setData(self.timestamps, self.data)
        self.save_data()
 ```  
![WhatsApp Image 2025-03-29 at 3 32 37 PM](https://github.com/user-attachments/assets/2f92c8d9-fb6a-4ce2-9b65-45f0baed4bc9)    
  |*Figura 2: Señal digitalizada (visualización dinamica).*| 

**☝️ ANALISIS- GRAFICO: ☝️**   
Con conocimiento previo del lenguaje de programación de python, se tienen en cuenta librearias como: sys, numpy, ctypes, csv, time, PyQt6.QtWidgets, PyQt6.QtCore, pyqtgraph, PyDAQmx, scipy.signal par el desarollo matemático y gráfico de la señal, luego de esto, teniendo en cuenta criterios teóricos que la frecuencia de una señal emg va hasta 500Hz. Establecimos la frecuencia de muestreo de 1000Hz cumpliendo el teorema de Nyquist, con el fin de obtener una señal mas límpia, aplicamos filtros digitales:   
▪️ 1. Filtro pasa altas en el cual establecimos una frecuencia de corte de 10Hz con el fin de eliminar ruido por movimiento.  
▪️ 2. Filtro pasa bajas estableciando frecuencia de corte de 500Hz  permitiendonos le paso de frecuencias correspondientes a la actividad muscular.    
Luego, se adquiere la señal por medio un sensor AD282, fue enviada a el ADQ6002 lo cual nos permite convertir la señal analógica a digital y enviarla por comunicación de manera digital permitiéndonos almacenarla en un buffer en donde se le aplican los respectivos filtros anteriormente mencionados, luego de esto se pasa a programar la gráfica de la señal, estableciendo eje de amplitud y tiempo, se programa para que sea mostrada en tiempo real con ayuda de self.timer.timeout.connect(self.update_plot) estamos generando una actualización de grafica cada 20ms haciendo que la grafica tenga un mejor flujo de señal, luego de este con ayuda de DATA_FILE = "emg_data.csv"  estamos generando un archivo csv con los datos de la señal que establecimos por un tiempo de 10 segundos, permitiéndonos llevar a cabo un estudio mas detallado de la señal. En la imagen se puede observar la señal EMG obtenida.   
___________________________________  
Ahora bien, dentro de la clase EMGPlot, se encuentra definida la función save_data(), encargada de guardar de manera continua los datos procesados de la señal EMG en un archivo .csv (Que nos enviara a un archivo en excel con todos los datos recolectados durante el tiempo de muestreo de la señal deseado). Esta función se ejecuta automáticamente cada vez que se actualiza la gráfica de la señal, es decir, aproximadamente cada 20 ml/seg. Su estructura interna comienza abriendo el archivo definido en la variable DATA_FILE (el cual es "emg_data.csv")=(# Archivo de almacenamiento) lo que significa que cada ejecución sobrescribe el contenido anterior del archivo, guardando únicamente los datos más recientes.

Luego, se utiliza la biblioteca csv para crear un escritor de archivos que primero añade una fila de encabezados: "Tiempo (s)" y "Voltaje (V)", que son los dos vectores generados durante la adquisición y el procesamiento de la señal. Posteriormente, mediante un ciclo for, se "procesan" de forma paralela los vectores self.timestamps (que contienen los tiempos en segundos) y self.data (que contiene los valores de voltaje filtrado), y cada par de valores se guarda en una nueva fila del archivo en el documento de **Excel**. De esta forma, cada fila del archivo representa una muestra puntual de la señal EMG filtrada en el tiempo, lo que nos permite un análisis más detallado sobre el comportamiento de nuestra señal adquirida.

```python
    def save_data(self):
        with open(DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Tiempo (s)", "Voltaje (V)"])
            for t, v in zip(self.timestamps, self.data):
                writer.writerow([t, v])
 ```

Para Finalizar, tenemos **closeEvent(self, event)** , que lo que hace es que asegura que cuando se cierra la ventana, se detiene correctamente el temporizador y la tarea de la DAQ, siendo asi una parte impoortante para poder evitar errores al cerrar la aplicación y el ultimo es el punto de entrada principal del programa ya que crea la aplicación Qt, inicializa la interfaz de usuario y lanza el bucle de eventos que teniamos al inicio del programa. A partir de aquí, todo el proceso de adquisición y visualización de la señal EMG comienza y por ello es efectiva la complilación de nuestro codigo.

```python
    def closeEvent(self, event):
        self.timer.stop()
        self.task.StopTask()
        self.task.ClearTask()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EMGPlot()
    window.show()
    sys.exit(app.exec())
 ```

## Conclusión: ⚙️ 


___________________________________     

## Licencia 
Open Data Commons Attribution License v1.0

## Temas:
# 📡 Procesamiento de Señales  
- 

# 🔊 Análisis en Frecuencia  
- 
- 

# 🖥️ Código e Implementación  
- Explicación del código  
- Ejecución y ejemplos  
- Mejoras y optimización  

              

