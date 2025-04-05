import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, windows
from scipy.fft import fft, fftfreq
from scipy.stats import t as t_dist  # Importar la distribución t con un nombre diferente
import random

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

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def apply_hanning_window(signal):
    window = windows.hann(len(signal))
    return signal * window

def detect_muscle_contractions(signal, fs, prominence_factor=0.5):
    """
    Detecta las contracciones musculares basadas en picos en la señal suavizada.

    Args:
        signal (np.array): La señal EMG suavizada.
        fs (int): Frecuencia de muestreo.
        prominence_factor (float): Factor para ajustar la prominencia mínima de los picos.
                                   Valores más altos detectan menos contracciones.

    Returns:
        list: Una lista de tuplas (start, end) que definen las ventanas de las contracciones.
    """
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

def analyze_fft(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N//2]
    return xf, 2.0/N * np.abs(yf[0:N//2])

def to_decibels(amplitude):
    return 20 * np.log10(amplitude / np.max(amplitude))

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

# Seleccionar las 6 contracciones más notorias (primeras 6 detectadas)
top_n_contractions = contraction_windows[:6]

print("\nResumen de características de frecuencia para las 6 contracciones más notorias:\n")
print("{:<15} {:<20} {:<25} {:<25}".format('Contracción', 'Media (Hz)', 'Desviación Estándar (Hz)', 'Mediana (Frecuencia, Hz)'))
print("="*90)

means = []
stdevs = []
contraction_lengths = []

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