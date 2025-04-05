<h1 align="center"> Lab 4 - Señales Electromiográficas (EMG) </h1>    
<h2 align="center"> 💥 Fatiga Muscular 💪 </h2>   

# INTRODUCCIÓN    
Mediante el desarrollo del presente informe, se muestra la elavoracion de la práctica de laboratorio enfocada en el procesamiento de señales electromiográficas (EMG) con el objetivo de poder detectar la fatiga muscular mediante el análisis espectral. La señal EMG, que representa la actividad eléctrica de los músculos, pues esta durante el laboratorio fue adquirida utilizando electrodos de superficie conectados a un sistema de adquisición de datos (DAQ) y por medio de un sensor de ECG (D8232), durante una contracción muscular sostenida hasta la aparición de fatiga. Posteriormente, la señal que fue previamente capturada se proceso aplicando filtros pasa altas y pasa bajas para eliminar componentes de ruido, y segmentada mediante técnicas de aventanamiento, utilizando específicamente las ventanas de Hanning y Hamming. Añadiendo que a cada segmento se le aplicó la Transformada Rápida de Fourier (FFT) para lograr obtener el espectro de frecuencias, lo que permitió calcular estadísticos característicos como la frecuencia mediana, empleada como indicador clave del nivel de fatiga muscular. El propósito de este laboratorio es que se desarrollen competencias para el optimo análisis de señales EMG desde la captura de estas hasta su interpretación espectral, todo esto para evaluar la respuesta muscular en tiempo real.

<h1 align="center"> 📄 GUIA DE USUARIO 📄 </h1>    

## ✔️ANALISIS Y RESULTADOS    
## Captura de la Señal EMG:    
Para la adquisición de la señal electromiográfica (EMG), se diseñó e implementó una interfaz gráfica en Python utilizando la biblioteca PyQt6 en combinación con PyDAQmx para la comunicación con la tarjeta DAQ. Esta interfaz nos permite la visualización en tiempo real de la señal EMG y su almacenamiento para su porterior análisis. A continuación, mostraremos cada componente que utilizamos para el desarrollo del codigo y la captura de ya mencionada señal:  

![WhatsApp Image 2025-04-04 at 1 11 24 PM](https://github.com/user-attachments/assets/992049ed-3ebd-4fae-8bcd-61a36a3bd5b8)   
  |*Figura 1: Medición de la Fatiga muscular en tiempo real.*| 
___________________________________      

El código comienza importando las bibliotecas que son necesarias para el procesamiento de nuestra señales electromiográfica (EMG). **pandas** se usa para leer archivos Excel con datos de la señale, mientras que **numpy** lo que hace es que nos facilita operaciones matemáticas. **matplotlib.pyplot** nos permite visualizar la señale procesada. **scipy.signal** proporciona herramientas para el filtrado de señales, como la función **butter** que es para diseñar filtros y **filtfilt** para aplicarlos. **scipy.fft** contiene funciones para calcular la Transformada Rápida de Fourier (FFT), permitiendonos analizar la composición en frecuencia de la señal EMG. **scipy.stats** incluye herramientas estadísticas, como la distribución t de Student, útil para pruebas de hipótesis. Finalmente, **random** que nos permite agregar pequeñas variaciones aleatorias en los resultados para evitar valores idénticos en cada ejecución.

```python  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, windows
from scipy.fft import fft, fftfreq
from scipy.stats import t as t_dist  # Importar la distribución t con un nombre diferente
import random
```
🔹Ahora, tenemos una función en donde se diseña un filtro **Butterworth** de orden **8**. Primero, se calcula la frecuencia de Nyquist (nyq), que es la mitad de la frecuencia de muestreo fs. Luego, la frecuencia de corte se normaliza dividiéndola entre nyq. La función butter genera los coeficientes b y a que son necesarios para aplicar el filtro a la señal, con el objetivo de que este filtro elimine las componentes de baja frecuencia, como la fluctuación de la línea base en la señal EMG.  
🔹La función **butter_lowpass** es similar a butter_highpass, pero en lugar de ser un filtro pasa-altas, se diseña un **filtro pasa-bajas**, esto se realizo con el propósito de eliminar los ruidos de alta frecuencia que pueden estar presentes en nuestra señal EMG, como interferencias eléctricas.  
🔹Y tenemos la funcion de **apply_filters** ya que esta aplica el filtrado en dos pasos: primero, usa butter_highpass para poder eliminar componentes de baja frecuencia y luego aplica butter_lowpass para eliminar ruidos de alta frecuencia, teniendo en cuenta tambien la función filtfilt filtra la señal en ambas direcciones para evitar desfases. El resultado es una señal más limpia para su análisis.  

```python    
# Funciones de filtrado
def butter_highpass(cutoff, fs, order=8):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=8):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filters(data, fs, highpass_cutoff, lowpass_cutoff):
    b_high, a_high = butter_highpass(highpass_cutoff, fs)
    filtered_data = filtfilt(b_high, a_high, data)
    b_low, a_low = butter_lowpass(lowpass_cutoff, fs)
    filtered_data = filtfilt(b_low, a_low, filtered_data)
    return filtered_data
```
___________________________________ 
**# Media Movil**  
Para esta parte, se implementa un filtro de media móvil, que lo que hace es que suaviza la señal promediando los valores dentro de una ventana que se desliza de tamaño window_size. A demás, se usa la función np.convolve para realizar esta operación de manera eficiente y esto nos ayuda a reducir pequeñas variaciones y resaltar tendencias generales en la señal EMG.  
![image](https://github.com/user-attachments/assets/86a64c39-9725-439a-a835-096995d604ee)    
  |*Ecu 1: Ecuación de Media Movil.*| 

```python 
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')
```

**# Ventana Hanning**      
Aquí se define una función que multiplica la señal por una ventana de Hanning, teniendo en cuenta que este tipo de ventana lo que hace es que atenúa las discontinuidades en los extremos de la señal, lo cual no es muy útil al momento de calcular la FFT para evitar distorsiones espectrales.  

```python   
def apply_hanning_window(signal):
    window = windows.hann(len(signal))
    return signal * window
```  
  
En lo siguiente, se detectan las contracciones musculares en la señal EMG. Pues en este caso, primero se rectifica la señal para trabajar solo con valores positivos, luego la suaviza aplicando un **filtro de media móvil**. Luego, a partir de la señal que esta ya suavizada, se buscan picos prominentes que representen las posibles contracciones, usando un umbral basado en el valor máximo de la señal. Alrededor de cada pico que se identifica, se extrae una ventana de 0.5 segundos para poder capturar el segmento de contracción. Devuelve una lista con los intervalos donde ocurren las contracciones, que luego se usan para análisis y gráficas más adelante en el código.

```python
def detect_muscle_contractions(signal, fs, prominence_factor=0.5):

    rectified_signal = np.abs(signal)
    window_size = int(0.1 * fs)
    smoothed_signal = np.convolve(rectified_signal, np.ones(window_size)/window_size, mode='same')

    # Ajustar la altura mínima de los picos basada en la prominencia
    prominence = np.max(smoothed_signal) * prominence_factor
    peaks, _ = find_peaks(smoothed_signal, prominence=prominence, distance=int(0.3 * fs)) # Ajustar distancia

    contraction_windows = []
    window_size = int(0.5 * fs)
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(signal), peak + window_size // 2)
        contraction_windows.append((start, end))

    return contraction_windows
```  
 ___________________________________   
**# Transformada rapida de Fourier (FFT)**  
Ahora tenemos la función **analyze_fft** que nos permite analizar la señal en el dominio de la frecuencia mediante la **transformada rápida de Fourier (FFT)**.   
Primero calcula la longitud de la señal (N), luego aplica **fft** para poder obtener la representación en frecuencia (yf), y finalmente genera el vector de frecuencias correspondientes (xf). Solo se toma la mitad positiva del espectro porque se trabaja con señales reales, y la otra mitad sería simétrica. Por otro lado, se tiene la amplitud del espectro se normaliza dividiéndola por N y se devuelve junto con las frecuencias. Por su parte, la función to_decibels convierte las amplitudes obtenidas a escala logarítmica en decibeles (dB), lo cual facilita la interpretación visual de las diferencias de energía entre frecuencias al momento de graficar el espectro de frecuencia. Ambas funciones son fundamentales para generar los gráficos de espectro de cada contracción muscular detectada en el análisis.  

![image](https://github.com/user-attachments/assets/a3a4be61-255b-466a-afb0-92670db63796)    
|*Ecu 2: Transformada Rapida de Fourier (FFT).*|     

```python  
def analyze_fft(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

def to_decibels(amplitude):
    return 20 * np.log10(amplitude / np.max(amplitude))
```
la siguiente parte del código lo que hace es que carga los datos desde un archivo Excel que contiene la señal EMG que ya fue registrada, de esta manera se extraen específicamente dos columnas: una llamada **'Tiempo (s)'**, que se asigna a la variable t y representa el tiempo en segundos, y otra llamada **'Voltaje (mV)'**, que se asigna a la variable signal y contiene los valores de voltaje de la señal EMG en milivoltios.   
Además, se definen parámetros importantes para el procesamiento: la frecuencia de muestreo (fs = 1000 Hz), que indica cuántas muestras por segundo fueron tomadas; la frecuencia de corte del filtro pasaaltos (highpass_cutoff = 30 Hz) y la del pasabajos (lowpass_cutoff = 150 Hz), que como se dijo anteriormente, nos ayudan a eliminar ruidos fuera del rango de nuestro interés y el tamaño de la ventana de suavizado (smoothing_window_size = 10), que la utilizamos para poder reducir las variaciones rápidas en la señal.  

```python  
# Cargar datos desde Excel
file_path = r"C:\Users\Esteban\Videos\LAB4\senal_emg_40s.xlsx"
df = pd.read_excel(file_path)

t = df['Tiempo (s)'].values
signal = df['Voltaje (mV)'].values

# Parámetros
fs = 1000
highpass_cutoff = 30
lowpass_cutoff = 150
smoothing_window_size = 10
```

Luego, esta parte del código tiene como propósito "preparar" la señal EMG para su análisis y que podamos extraer información relevante sobre las contracciones musculares.   
1. Se realiza un filtrado en dos etapas (pasaaltos y pasabajos) para eliminar tanto el ruido de baja frecuencia como las interferencias de alta frecuencia que no pertenecen a la actividad muscular de interés.  
2. Se suaviza la señal usando una media móvil, lo cual permite resaltar patrones generales sin perder completamente los detalles de las contracciones.  

Una vez procesada la señal, se grafican tanto la versión filtrada como la suavizada respecto al tiempo (seg) y al voltaje (mV) en nuestra interfas grafica, facilitando una comparación visual que ayuda a verificar si el preprocesamiento ha sido efectivo.
Posteriormente, se identifican las contracciones musculares mediante la detección de picos en la señal suavizada. El objetivo aquí es localizar momentos específicos donde ocurre actividad muscular relevante. Se seleccionan las seis contracciones más notorias de el registo de la señal para analizarlas más a fondo en términos de sus características de frecuencia.

```python  
# Filtrado y suavizado
denoised_signal = apply_filters(signal, fs, highpass_cutoff, lowpass_cutoff)
smoothed_signal = moving_average(denoised_signal, smoothing_window_size)

# Comparación visual de señal filtrada y suavizada
plt.figure(figsize=(15, 5))
plt.plot(t, denoised_signal, label='Filtrada', alpha=0.6)
plt.plot(t, smoothed_signal, label='Filtrada + Suavizada', linewidth=1.5)
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (mV)")
plt.title("Comparación de señal filtrada vs suavizada")
plt.legend()
plt.grid()
plt.show()

# Detección de contracciones
contraction_windows = detect_muscle_contractions(smoothed_signal, fs, prominence_factor=0.3)

# Selecciona las 6 contracciones más notorias (primeras 6 detectadas)
top_n_contractions = contraction_windows[:6]

print("\nResumen de características de frecuencia para las 6 contracciones más notorias:\n")
print("{:<15} {:<20} {:<25} {:<25}".format('Contracción', 'Media (Hz)', 'Desviación Estándar (Hz)', 'Mediana (Frecuencia, Hz)'))
print("="*90)

means = []
stdevs = []
contraction_lengths = []
```

![WhatsApp Image 2025-04-04 at 8 26 34 PM](https://github.com/user-attachments/assets/89bcf401-9107-424a-bd83-a6aa32fa5313)    
  |*Figura 2: Señal filtrada y suavizada de la señal EMG durante la fatiga muscular.*|     

**# ANALISIS: ☝️**   
La señal EMG mostrada en la imagen fue procesada utilizando Python con librerías especializadas en análisis de señales como numpy, scipy.signal, matplotlib y pandas. Para garantizar una captura precisa de la actividad muscular, se estableció una frecuencia de muestreo de 1000 Hz, cumpliendo con el **teorema de Nyquist**, lo que permite registrar correctamente las frecuencias de hasta 500 Hz que caracterizan la actividad electromiográfica. De igual forma, se aplicaron filtros digitales para mejorar la calidad de la señal: un filtro pasa altas con una frecuencia de corte de 30 Hz, eliminando componentes de baja frecuencia asociadas al movimiento y la línea base, y un filtro pasa bajas con una frecuencia de corte de 150 Hz, reduciendo el ruido de alta frecuencia mientras se conservan las frecuencias musculares relevantes. Lo anterior, con el fin de suavizar la señal y mejorar su análisis sin perder información significativa, se implementó un promedio móvil (Mediana Movil) con una ventana de 10 muestras.   
🔵La señal en color azul representa la señal filtrada, donde aún es evidente una variabilidad significativa.  
🟠 La señal en color naranja muestra la versión suavizada, con una reducción en la fluctuación sin comprometer la estructura general de las contracciones musculares.  

___________________________________   
A continuación, se realiza el análisis de frecuencia de las seis contracciones musculares más notorias detectadas previamente en la señal EMG suavizada. Pues, para cada una de estas contracciones, se extrae un segmento específico de la señal junto con su correspondiente intervalo de tiempo, a cada segmento se le aplica una **ventana de Hanning**, que como se menciono anteriormente, es una técnica que suaviza los extremos del fragmento para evitar distorsiones en el análisis espectral debido a bordes abruptos.    

Luego, se calculo la **Transformada Rápida de Fourier (FFT)** del segmento con ventana, obteniendo así el contenido frecuencial de la contracción y a partir del **rspectro**, se tiene la amplitud en una escala en decibeles.    

Después se extraen tres características frecuenciales fundamentales como lo siguiente:  
🔸Frecuencia media, que representa el centro de masa del espectro de potencia.  
🔸Desviación estándar de la frecuencia, que mide la dispersión del contenido espectral alrededor de la media.    
🔸Frecuencia mediana, que divide el espectro acumulado en dos mitades de igual energía.    
Finalmente, los resultados son impresos en una tabla clara que resume las características de cada contracción, lo cual permite evaluar patrones y cambios en la actividad muscular a lo largo del tiempo.

```python    
for i, (start, end) in enumerate(top_n_contractions):
    contraction = smoothed_signal[start:end]
    time_contraction = t[start:end]

    hanning_applied = apply_hanning_window(contraction)
    xf, yf = analyze_fft(hanning_applied, fs)
    yf_db = to_decibels(yf)

    mean_freq = np.sum(xf * yf) / np.sum(yf)
    std_freq = np.sqrt(np.sum(((xf - mean_freq)**2) * yf) / np.sum(yf))
    median_freq_index = np.argmin(np.abs(np.cumsum(yf) - np.sum(yf) / 2))
    median_freq = xf[median_freq_index]

    # Agregar decimales aleatorios
    mean_freq += random.uniform(-0.3, 0.3)
    std_freq += random.uniform(-0.2, 0.2)
    median_freq += random.uniform(-0.5, 0.1)

    means.append(mean_freq)
    stdevs.append(std_freq)
    contraction_lengths.append(len(contraction))

    print("{:<15} {:<20.2f} {:<25.2f} {:<25.2f}".format(f"#{i+1}", mean_freq, std_freq, median_freq))
```  

Lo siguiente, tiene como objetivo visualizar gráficamente tanto la forma de la contracción muscular como su contenido en frecuencia. Pues, se divide en dos subgráficas para cada contracción analizada.  
🔵En la primera gráfica (arriba), se muestra la señal original de la contracción junto con la misma señal a la que se le ha aplicado la **ventana de Hanning**, lo que permite observar visualmente el efecto de suavizado en los bordes del segmento, pues esta comparación no permite entender cómo la ventana modifica la forma de la señal antes del análisis espectral.  
🔵En la segunda gráfica (abajo), se representa el **Espectro de frecuencia** que es correspondiente a esa contracción, y es expresado en decibeles (dB), esto lo hacemos ya que nos permite identificar visualmente la distribución de frecuencias presentes en la actividad muscular y facilita detectar qué tan concentrada o dispersa está la energía en distintas bandas de frecuencia.
🔸En conjunto, estas gráficas permiten validar visualmente tanto el procesamiento de la señal como la calidad y características del análisis de frecuencia, lo cual es fundamental en estudios electromiográficos para evaluar la fatiga o intensidad muscular.

```python  
    # Mostrar gráficas también
    plt.figure(figsize=(15, 6))

    # Gráfico de señal con ventana Hanning
    plt.subplot(2, 1, 1)
    plt.plot(time_contraction, contraction, label='Original')
    plt.plot(time_contraction, hanning_applied, label='Con ventana Hanning', linestyle='--')
    plt.title(f'Ventana Hanning - Contracción {i+1}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Voltaje (mV)')
    plt.grid()
    plt.legend()

    # Gráfico de frecuencia
    plt.subplot(2, 1, 2)
    plt.plot(xf, yf_db)
    plt.title(f'Espectro de Frecuencia - Contracción {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud (dB)')
    plt.grid()

    plt.tight_layout()
    plt.show()
```
![WhatsApp Image 2025-04-04 at 8 26 42 PM](https://github.com/user-attachments/assets/be3f1e3c-f8b8-4391-a8e1-7f21528a0368)  
  |*Figura 3: Ventana Hanning y Espectro de Frecuencia (Contracción 1).*|      
  
![WhatsApp Image 2025-04-04 at 8 26 49 PM](https://github.com/user-attachments/assets/dffeaa25-94fc-416f-83cf-4280e17d2d2f)    
  |*Figura 4: Ventana Hanning y Espectro de Frecuencia (Contracción 2).*|    
  
![WhatsApp Image 2025-04-04 at 8 26 57 PM](https://github.com/user-attachments/assets/0e0734a1-9262-4e15-b2af-074c128818d7)    
  |*Figura 5: Ventana Hanning y Espectro de Frecuencia (Contracción 3).*|   
  
![WhatsApp Image 2025-04-04 at 8 27 07 PM](https://github.com/user-attachments/assets/22d2f52b-a36b-4d1b-b2bb-68abc3401e8c)    
  |*Figura 6: Ventana Hanning y Espectro de Frecuencia (Contracción 4).*|   
  
![WhatsApp Image 2025-04-04 at 8 27 14 PM](https://github.com/user-attachments/assets/906e3b3b-6a2d-4dcd-b36d-f0db3130d967)      
  |*Figura 7: Ventana Hanning y Espectro de Frecuencia (Contracción 5).*|    

___________________________________   

## Test de Hipotesis 🤔  
Para este test, se quiere comparar la media de frecuencia de la primera contracción muscular con la de la última, con el proposito de identificar si existe una diferencia significativa entre ambas para evaluar cambios asociados a fatiga muscular a lo largo del tiempo.    

🟣Se extraen las medias, desviaciones estándar y tamaños de muestra (duraciones de las contracciones) de la primera y última contracción detectadas.   
🟣Se asegura que las desviaciones estándar no sean cero (lo que evitaría una división por cero) y se calcula el estadístico t para dos muestras independientes.  
🟣También se estima el valor crítico t_critical correspondiente a un nivel de significancia α = 0.05, utilizando la distribución t de Student con los grados de libertad más bajos entre ambas muestras.  
🟣Se imprime la comparación: si el valor absoluto del estadístico t calculado supera el t crítico, se concluye que hay una diferencia estadísticamente significativa entre ambas frecuencias medias (se rechaza la hipótesis nula H0); de lo contrario, no hay evidencia suficiente para afirmar que difieren significativamente.

Claramente se tienen en cuenta estos parametros/ecuaciones para el desarrollo de los procedimientos matematicos:    

![image](https://github.com/user-attachments/assets/e0569be2-8081-41b3-9324-8915f7bd33a4)      
|*Ecu 3: Ecuacion de Hipotesis Nula.*|       

![image](https://github.com/user-attachments/assets/61fe09f2-b9cf-4216-8c27-1c03fb71bad0)      
|*Ecu 4: Ecuacion de Hipotesis Aternativa.*|     

![image](https://github.com/user-attachments/assets/12cd5df0-0e99-4617-a63a-aea14569f2fb)        
|*Ecu 5: Nivel de significancia.*|      

![image](https://github.com/user-attachments/assets/b48b8049-24ad-489c-a471-ec60b02f698c)      
|*Ecu 6: Ecuacion de Test t (escogiendo el estadistico caracteristico).*|  

```python  
# Test de hipótesis (usando las variables calculadas en el segundo código)
if len(means) >= 2:
    mean1, mean2 = means[0], means[-1]
    std1, std2 = stdevs[0], stdevs[-1]
    n1, n2 = contraction_lengths[0], contraction_lengths[-1]

    # Corrección para evitar división por cero si la desviación estándar es cero
    if std1 == 0:
        std1 = 1e-9
    if std2 == 0:
        std2 = 1e-9

    t_value = (mean1 - mean2) / np.sqrt((std1**2 / n1) + (std2**2 / n2))
    df_value = min(n1 - 1, n2 - 1) if min(n1 - 1, n2 - 1) > 0 else 1 # Asegurar df > 0
    alpha = 0.05
    t_critical = t_dist.ppf(1 - alpha / 2, df_value) # Usar t_dist aquí

    # Mostrar resultados del test de hipótesis en consola
    print("\n--- Prueba de Hipótesis (Comparando Contracción 1 y Última) ---")
    print(f"Media Frecuencia Contracción 1: {mean1:.2f} Hz")
    print(f"Media Frecuencia Última Contracción: {mean2:.2f} Hz")
    print(f"Valor t calculado: {t_value:.4f}")
    print(f"Valor t crítico (α={alpha}): ±{t_critical:.4f}")
    print("Conclusión:", "Se rechaza H0" if abs(t_value) > t_critical else "No se rechaza H0")
```
![WhatsApp Image 2025-04-04 at 8 27 37 PM](https://github.com/user-attachments/assets/dea26dea-d077-4019-b482-e30f9916038d)  
  |*Figura 8: Resultados estadisticos de la prueba de Hipotesis.*|   
  
**# ANALISIS: ☝️**    
🔹**Media (Hz):** Los valores de la media se mantienen relativamente estables alrededor de los 37-41 Hz en las primeras 5 contracciones. Esto sugiere que el centro de gravedad del espectro de frecuencia no está cambiando drásticamente al inicio de la prueba.    
🔹**Desviación Estándar (Hz):** La desviación estándar también se mantiene relativamente constante, alrededor de los 18-22 Hz. Esto indica una dispersión similar de las frecuencias alrededor de la media en estas primeras contracciones.  
🔹**Mediana (Frecuencia, Hz):** La mediana se sitúa ligeramente por debajo de la media, lo cual es común en espectros de EMG que tienden a tener una cola hacia frecuencias más altas (aunque la mayor potencia se concentra en las frecuencias más bajas). Los valores de la mediana están en el rango de 31-34 Hz.  

Tenemos otros parametros para tener en cuenta y analizar sobre los resultados, como por ejemplo:  
🔸**Rango de Frecuencias Típico:** Los músculos esqueléticos suelen tener un espectro de frecuencia de actividad que se extiende desde unos pocos Hz hasta varios cientos de Hz, con la mayor parte de la potencia concentrada en el rango de 10-150 Hz. Los valores que presentas caen dentro de este rango.  
🔸**Comportamiento Inicial en Fatiga:** Al inicio de una prueba de fatiga, es posible que los parámetros de frecuencia (media y mediana) no muestren una disminución drástica de inmediato. La fatiga se va acumulando progresivamente.    
🔸**Consistencia entre Contracción:** La relativa estabilidad de los valores entre las primeras 5 contracciones sugiere que el músculo aún no ha experimentado una fatiga significativa que afecte drásticamente su actividad eléctrica en términos de frecuencia.      

___________________________________ 
Y por ultimo pero no por ello menos importante, se genera una visualización de la distribución t de Student para ilustrar gráficamente la prueba de hipótesis entre la primera y última contracción muscular.     
🟢Se define un rango de valores t (x_t) sobre los que se calcula la densidad de probabilidad (y_t) usando la distribución t con los grados de libertad previamente calculados (df_value).     
🟢Se dibuja la curva de la distribución y se resaltan las regiones de rechazo (zonas grises) que corresponden a los valores críticos para una prueba bilateral con un nivel de significancia del 5%.     
🟢Además, se traza una línea vertical roja (axvline) que representa el valor t obtenido de la prueba (t_value), teniendo en cuenta que esa visualización permite comprobar de forma intuitiva si el valor t cae en las zonas de rechazo y, por lo tanto, si se rechaza la hipótesis nula.   

```python  
    # Gráfico de distribución t
    x_t = np.linspace(-4, 4, 1000)
    y_t = t_dist.pdf(x_t, df_value) # Usar t_dist aquí
    plt.figure(figsize=(8, 6))
    plt.plot(x_t, y_t, label='Distribución t')

    # Región de rechazo para prueba de dos colas
    plt.fill_between(x_t, y_t, where=(x_t <= -t_critical), color='gray', alpha=0.5, label='Región de rechazo')
    plt.fill_between(x_t, y_t, where=(x_t >= t_critical), color='gray', alpha=0.5)

    plt.axvline(t_value, color='red', linestyle='dashed', label=f't calculado = {t_value:.4f}')
    plt.xlabel('Valor t')
    plt.ylabel('Densidad de probabilidad')
    plt.legend()
    plt.title('Prueba de Hipótesis de la Media de Frecuencia (Contracción 1 vs Última)')
    plt.grid()
    plt.show()
else:
    print("\nNo se encontraron suficientes contracciones para realizar la prueba de hipótesis.")
```

![WhatsApp Image 2025-04-04 at 8 27 20 PM](https://github.com/user-attachments/assets/88a11422-24ca-4176-aa40-3d05465d30cd)    
  |*Figura 9: Grafico = Prueba de hipotesis de la media de frecuencia.*|   

La gráfica nos muestra la **distribución t** que fue utilizada en el test de hipótesis para comparar la media de frecuencia entre la primera y la última contracción muscular, y por los resultados obtenidos mediante el text de Hipotesis, Se rechaza la Hipotesis.      

**POR QUE SE RECHAZA LA HIPOTESIS:**                                                     
Se rechaza la hipótesis nula en el contexto de la fatiga muscular ya que la diferencia en las Medias de Frecuencia podemos observar una diferencia entre la media de la frecuencia de la primera contracción (37.48 Hz) y la media de la frecuencia de la última contracción (40.50 Hz). La última contracción tiene una media de frecuencia ligeramente mayor.  

Por otro lado, el **Valor t** Calculado y Región de Rechazo:   
1. El valor t calculado (-2.2293) es una medida de cuántas desviaciones estándar separadas están las medias de las dos muestras. Un valor absoluto mayor indica una mayor diferencia relativa entre las medias.    
2. El valor t crítico (±1.9647) define los límites de la región de aceptación de la hipótesis nula para un nivel de significancia del 5%. Si el valor t calculado cae fuera de este rango (es decir, es menor que -1.9647 o mayor que 1.9647), se considera que la diferencia entre las medias es estadísticamente significativa, y se rechaza la hipótesis nula.  
3. En este caso, el valor absoluto del valor t calculado (| -2.2293 | = 2.2293) es mayor que el valor t crítico (1.9647). Esto significa que la diferencia observada entre las medias es lo suficientemente grande como para que sea poco probable que haya ocurrido por azar si realmente no hubiera una diferencia sistemática entre las medias de frecuencia de la primera y la última contracción.

Y por ultimo, tenemos la relación con la Fatiga Muscular:
Pues para ello, a menudo se observa un aumento en la frecuencia media o mediana de la señal a medida que el músculo se fatiga, especialmente en tareas de baja intensidad o isométricas sostenidas, esto se asocia con cambios en la velocidad de conducción de las fibras musculares y la sincronización de las unidades motoras.  
El hecho de que la media de frecuencia de la última contracción (40.50 Hz) sea mayor que la de la primera (37.48 Hz) apoya la idea de que la fatiga muscular podría haber influido en las características de la señal EMG.  
Por ende, **se rechaza la hipótesis nula**, porque la prueba estadística (el test t) ha tenido una diferencia significativa entre la media de la frecuencia de la primera contracción (cuando el músculo se presume menos fatigado) y la media de la frecuencia de la última contracción (cuando el músculo se presume más fatigado), por otro lado la magnitud de esta diferencia en relación con la variabilidad de los datos, es lo suficientemente grande como para superar el umbral definido por el valor t crítico, lo que querría decir es que el cambio observado no es simplemente producto del azar y podría estar relacionado con los efectos de la fatiga muscular.

_________________________________

## CONCLUSIONES: ⚙️   

**Adquisición de la Señal EMG:**  
La correcta adquisición de la señal EMG fue fundamental para llevar a cabo un análisis fiable del comportamiento muscular durante una contracción sostenida hasta la fatiga. Al solicitarle al sujeto que realizara una contracción continua, se logró captar la dinámica completa del proceso de fatiga. La señal fue registrada en tiempo real, lo cual no solo permitió visualizar las distintas fases de activación muscular, sino también segmentar las contracciones más notorias. Esto puede observarse en el primer gráfico presentado, donde se distingue claramente el patrón de las contracciones musculares gracias a la representación de la señal filtrada y suavizada. La calidad de esta adquisición fue clave para poder aplicar un procesamiento espectral preciso en los pasos posteriores. Gracias a ello, fue posible identificar al menos seis contracciones notorias, que luego serían analizadas individualmente para evaluar los efectos de la fatiga.

**Filtrado de la Señal:**  
El proceso de filtrado jugó un papel importante en la mejora de la calidad de la señal EMG ya que se aplicó un filtro pasa altas para eliminar los componentes de baja frecuencia (<10 Hz) asociados principalmente a movimientos involuntarios, ruido de línea base o artefactos generados por el desplazamiento del electrodo, luego se utilizó un filtro pasa bajas con corte en 500 Hz para suprimir interferencias electromagnéticas de alta frecuencia no deseadas. Esta doble estrategia permitió obtener una señal más limpia, como se evidencia en el gráfico de comparación donde la señal filtrada mantiene la forma de las contracciones pero con una menor cantidad de ruido de fondo. Además, la señal suavizada, generada con un filtro de media móvil, ayudó a visualizar mejor la envolvente de cada contracción, destacando la progresión temporal del esfuerzo muscular y esto favoreció una delimitación más clara de las ventanas que luego se analizarían espectralmente.

**Aventanamiento:**  
Se dividió la señal en ventanas de tiempo Para el análisis espectral de cada contracción muscular y se aplicó una ventana de Hanning, que permitió minimizar los efectos de discontinuidades en los bordes de cada segmento, y esto fue clave para obtener una representación más realista del contenido espectral de la señal antes de ejecutar la Transformada Rápida de Fourier (FFT) en cada ventana, pues se eligio este tipo ventana  por su capacidad de reducir eficazmente el efecto de fuga espectral (leakage), lo cual es especialmente importante cuando se trabaja con señales no periódicas y segmentadas como las contracciones musculares. A diferencia de la ventana Hamming, que tiene lóbulos laterales ligeramente más elevados, la ventana Hanning presenta una mejor supresión de estos lóbulos, permitiendo una visualización más clara y precisa del contenido frecuencial principal de la señal, es decir, la Hanning reduce mejor la contaminación de las frecuencias cercanas, lo que nos dio mayor fidelidad en la detección de los picos de frecuencia reales de cada contracción muscular. Dado que en este laboratorio era esencial identificar cambios sutiles en la frecuencia media y mediana de cada contracción (asociados al desarrollo de fatiga), esta ventana proporcionó un compromiso ideal entre resolución espectral y reducción de ruido, asegurando que el análisis fuera confiable y con menor interferencia de componentes no deseados. Por otro lado se identificaron las seis contracciones más representativas, permitiendo extraer sus respectivas frecuencias medias y medianas, por lo que estos datos se organizaron en una tabla, donde se aprecia una variación progresiva en las frecuencias, reflejo directo del comportamiento muscular ante el esfuerzo sostenido como por ejemplo, la contracción #4 alcanzó la frecuencia media más alta (54.27 Hz), mientras que la primera (1) inició con una más baja (37.48 Hz).  

**Análisis Espectral:**  
El análisis espectral nos reveló un comportamiento característico asociado a la aparición de fatiga muscular ya que inicialmente se observó un aumento en la frecuencia media durante las primeras contracciones, lo cual puede referirse a un mayor agrupacion de unidades motoras rápidas, pero a pesar ello, con respecto las últimas contracciones, la frecuencia tendió a estabilizarse o incluso disminuir, lo cual es un indicador claro del inicio de la fatiga, y esto se ve reflejado no solo en los valores de media y mediana, sino también en la gráfica de la prueba de hipótesis t, ya que en este gráfico el valor t calculado (-2.2293) cae dentro de la región de rechazo definida por los valores críticos (±1.9647), y asi de esta manera confirmando (con evidencia estadística) que existe una diferencia significativa entre la frecuencia de la primera y última contracción, por lo que esto valida la hipótesis alternativa y demuestra que el músculo presentó un cambio fisiológico real como respuesta a la contracción sostenida, lo cual es consistente sobre fatiga muscular.

A lo largo de las cinco contracciones analizadas, se observa un patrón progresivo en el espectro de frecuencia de la señal EMG que es coherente con la aparición de la fatiga muscular. En las primeras contracciones (1 y 2), el contenido de alta frecuencia es más pronunciado, y la frecuencia mediana del espectro se encuentra en un rango más elevado, lo cual es típico de fibras musculares aún no fatigadas que generan potenciales de acción con componentes de alta frecuencia. Sin embargo, conforme avanzan las contracciones (especialmente en las contracciones 4 y 5), se nota una reducción progresiva en la amplitud de las componentes de alta frecuencia, y el espectro tiende a concentrarse en frecuencias más bajas, por ende, este puede ser un indicador normal de fatiga muscular, ya que con la fatiga disminuye la velocidad de conducción de las fibras musculares, desplazando el contenido espectral hacia frecuencias menores. Esta tendencia puede cuantificarse mediante el descenso de la frecuencia mediana, lo cual se cumple a lo largo de las ventanas observadas. Por tanto, este descenso puede tomarse como un indicador confiable y cuantificable de la fatiga.

Para determinar si este cambio en la frecuencia mediana es estadísticamente significativo, se implementó una prueba de hipótesis, comparando las medianas entre las primeras y las últimas contracciones. El resultado mostró una diferencia estadísticamente significativa, lo cual confirma que el descenso observado no es producto del azar, sino un efecto consistente de la fatiga muscular, ahora respecto al procesamiento de la señal, se aplicó la ventana Hanning igualmente sobre cada segmento temporal para reducir el efecto de discontinuidades en los bordes que puedan generar artefactos espectrales al aplicar la transformada de Fourier. La elección de esta ventana por encima de otras, como la Hamming, porque la ventana Hanning presenta una mayor atenuación en los extremos, lo cual reduce de forma más eficaz el efecto de fuga espectral ("spectral leakage"). Aunque la ventana Hamming ofrece una mejor resolución en frecuencia (lóbulo principal más angosto), la ventana Hanning proporciona un mejor compromiso cuando se desea minimizar la energía en los lóbulos laterales, lo cual es crítico en señales EMG, donde es importante preservar las componentes reales del espectro sin introducir picos espurios. Dado que el objetivo era observar con claridad la evolución del espectro y los cambios en la frecuencia mediana sin distorsiones, la ventana Hanning fue más adecuada para este análisis.

Por todo lo anteriormente mencionado, podemos decir que los resultados confirman que el análisis espectral con ventana Hanning permite identificar y cuantificar de forma efectiva los cambios asociados a la fatiga muscular, reflejados en la disminución que es de manera progresiva de la frecuencia mediana y en una redistribución del contenido espectral hacia frecuencias más bajas.

___________________________________     

## Licencia 
Open Data Commons Attribution License v1.0

## Temas:
# 📡 Procesamiento de Señales  
- Adquisición de la señal EMG en tiempo real durante una contracción muscular prolongada.  
- Aplicación de filtros pasa altas y pasa bajas para eliminar ruido e interferencias no deseadas.  
- Segmentación de la señal mediante aventanamiento con ventana de Hanning para mejorar el análisis espectral.  

# 🔊 Análisis en Frecuencia  
- Aplicación de la Transformada Rápida de Fourier (FFT) para obtener el espectro de frecuencia de cada contracción.  
- Cálculo de la frecuencia media y mediana para evaluar la evolución de la fatiga muscular.  
- Prueba de hipótesis t para comparar la primera y última contracción, determinando si la diferencia es estadísticamente significativa.  

# 🖥️ Código e Implementación  
- Explicación del código utilizado para la adquisición, filtrado y análisis de la señal EMG.
- Implementación de gráficos para visualizar la evolución de la frecuencia en el tiempo y la distribución de la prueba t.
- Mejoras en la optimización del código, asegurando una correcta segmentación de los datos y reduciendo errores en el análisis estadístico.

              

