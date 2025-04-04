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

# Diseño del filtro Butterworth
b, a = butter(4, [HP_CUTOFF / (0.5 * FS), min(LP_CUTOFF / (0.5 * FS), 0.99)], btype='band', analog=False)

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
    
    def read_data(self):
        data = np.zeros(BUFFER_SIZE, dtype=np.float64)
        read = ctypes.c_int32(0)
        self.ReadAnalogF64(BUFFER_SIZE, 10.0, DAQmx_Val_GroupByChannel, data, BUFFER_SIZE, ctypes.byref(read), None)

        # Si no se leyeron suficientes muestras, devolver ceros para evitar ruido
        if read.value < BUFFER_SIZE:
            return np.zeros(BUFFER_SIZE)

        return filter_signal(data)

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

    def save_data(self):
        with open(DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Tiempo (s)", "Voltaje (V)"])
            for t, v in zip(self.timestamps, self.data):
                writer.writerow([t, v])

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


