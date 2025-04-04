<h1 align="center"> Lab 4 - Señales Electromiográficas (EMG) </h1>    
<h2 align="center"> 💥 Fatiga Muscular 💪 </h2>   

# INTRODUCCIÓN    
Mediante el desarrollo del presente informe, se muestra la elavoracion de la práctica de laboratorio enfocada en el procesamiento de señales electromiográficas (EMG) con el objetivo de poder detectar la fatiga muscular mediante el análisis espectral. La señal EMG, que representa la actividad eléctrica de los músculos, pues esta durante el laboratorio fue adquirida utilizando electrodos de superficie conectados a un sistema de adquisición de datos (DAQ), durante una contracción muscular sostenida hasta la aparición de fatiga. Posteriormente, la señal que fue previamente capturada se proceso aplicando filtros pasa altas y pasa bajas para eliminar componentes de ruido, y segmentada mediante técnicas de aventanamiento, utilizando específicamente las ventanas de Hanning y Hamming. Añadiendo que a cada segmento se le aplicó la Transformada Rápida de Fourier (FFT) para lograr obtener el espectro de frecuencias, lo que permitió calcular estadísticos característicos como la frecuencia mediana, empleada como indicador clave del nivel de fatiga muscular. El propósito de este laboratorio es que se desarrollen competencias para el optimo análisis de señales EMG desde la captura de estas hasta su interpretación espectral, todo esto para evaluar la respuesta muscular en tiempo real.

<h1 align="center"> 📄 GUIA DE USUARIO 📄 </h1>    

## ✔️ANALISIS Y RESULTADOS    
## Captura de la Señal EMG:    
Para la adquisición de la señal electromiográfica (EMG), se diseñó e implementó una interfaz gráfica en Python utilizando la biblioteca PyQt6 en combinación con PyDAQmx para la comunicación con la tarjeta DAQ. Esta interfaz nos permite la visualización en tiempo real de la señal EMG y su almacenamiento para su porterior análisis. A continuación, mostraremos cada componente que utilizamos para el desarrollo del codigo y ña captura de ya mencionada señal:
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


<img src="![WhatsApp Image 2025-04-04 at 1 11 24 PM](https://github.com/user-attachments/assets/d7475179-d754-4782-882a-fe4698398ef0)" width="350" height="400">   
  |*Figura 1: Medición de la Fatiga muscular en tiempo real.*|  

```python

 ```


              

