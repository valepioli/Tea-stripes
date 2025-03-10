import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import Button, filedialog, simpledialog
import matplotlib.pyplot as plt

# Path al video
video_path = r"C:\Users\Valeria Pioli\OneDrive\Desktop\IPT\videos\8g.mp4"

# Selezione directory di output tramite finestra di dialogo
def select_output_directory():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter
    output_dir = filedialog.askdirectory(title="Seleziona la cartella per salvare i risultati")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Finestra per inserire il nome del file
def get_file_name():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter
    file_name = simpledialog.askstring("Nome file", "Inserisci il nome del file di salvataggio:")
    if not file_name:
        print("Nome file non fornito. Uscita.")
        exit()
    return file_name

output_dir = select_output_directory()
if not output_dir:
    print("Directory di output non selezionata. Uscita.")
    exit()

file_name = get_file_name()

# Variabili globali per la selezione ovale
ellipse_center_main = None
ellipse_axes_main = None
ellipse_center_exclude = []
ellipse_axes_exclude = []
is_drawing = False
selection_done = False
current_selection = "main"  # Determina se si sta disegnando l'ellisse principale o quella di esclusione

def draw_ellipse(event, x, y, flags, param):
    global ellipse_center_main, ellipse_axes_main, ellipse_center_exclude, ellipse_axes_exclude, is_drawing, current_selection
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_selection == "main":  # Seleziona l'ellisse principale
            ellipse_center_main = (x, y)
            ellipse_axes_main = (0, 0)
        else:  # Seleziona l'ellisse da escludere
            ellipse_center_exclude.append((x, y))
            ellipse_axes_exclude.append((0, 0))
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        axes_x = abs(x - ellipse_center_main[0])
        axes_y = abs(y - ellipse_center_main[1])
        if current_selection == "main":
            ellipse_axes_main = (axes_x, axes_y)
        else:
            axes_x = abs(x - ellipse_center_exclude[-1][0])
            axes_y = abs(y - ellipse_center_exclude[-1][1])
            ellipse_axes_exclude[-1] = (axes_x, axes_y)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

def confirm_main_selection():
    global current_selection
    current_selection = "exclude"  # Dopo la selezione principale, passiamo alla selezione delle aree da escludere
    print("Selezione principale confermata, ora seleziona le aree da escludere.")

def confirm_exclusion_selection():
    global selection_done
    selection_done = True
    print("Selezione delle aree da escludere confermata.")
    root.destroy()

def manual_contour_selection(frame):
    global ellipse_center_main, ellipse_axes_main, ellipse_center_exclude, ellipse_axes_exclude, selection_done, current_selection
    selection_done = False
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    temp_frame = frame.copy()

    # Finestra di selezione con le proporzioni del video
    video_height, video_width = frame.shape[:2]
    aspect_ratio = video_width / video_height

    # Imposta la finestra in modo da mantenere l'aspect ratio del video
    window_width = 800
    window_height = int(window_width / aspect_ratio)

    cv2.namedWindow("Select Oval Area", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Oval Area", window_width, window_height)
    cv2.setMouseCallback("Select Oval Area", draw_ellipse)

    global root
    root = tk.Tk()
    root.title("Confirm Selection")
    root.geometry("200x100")

    # Pulsante per confermare la selezione principale
    Button(root, text="OK (Main Selection)", command=confirm_main_selection).pack()

    while current_selection == "main" and not selection_done:
        temp_frame = frame.copy()
        if ellipse_center_main and ellipse_axes_main:
            cv2.ellipse(temp_frame, ellipse_center_main, ellipse_axes_main, 0, 0, 360, (0, 255, 0), 2)
        cv2.imshow("Select Oval Area", temp_frame)
        cv2.waitKey(1)
        root.update()

    # Pulsante per confermare la selezione di esclusione
    Button(root, text="OK (Exclusion Selection)", command=confirm_exclusion_selection).pack()

    while current_selection == "exclude" and not selection_done:
        temp_frame = frame.copy()
        if ellipse_center_main and ellipse_axes_main:
            cv2.ellipse(temp_frame, ellipse_center_main, ellipse_axes_main, 0, 0, 360, (0, 255, 0), 2)
        # Disegna le ellissi da escludere in rosso
        for i in range(len(ellipse_center_exclude)):
            cv2.ellipse(temp_frame, ellipse_center_exclude[i], ellipse_axes_exclude[i], 0, 0, 360, (0, 0, 255), 2)
        cv2.imshow("Select Oval Area", temp_frame)
        cv2.waitKey(1)
        root.update()

    # Crea la maschera finale
    if ellipse_center_main and ellipse_axes_main:
        cv2.ellipse(mask, ellipse_center_main, ellipse_axes_main, 0, 0, 360, 255, -1)
    for i in range(len(ellipse_center_exclude)):
        if ellipse_center_exclude[i] and ellipse_axes_exclude[i]:
            cv2.ellipse(mask, ellipse_center_exclude[i], ellipse_axes_exclude[i], 0, 0, 360, 0, -1)

    cv2.destroyAllWindows()
    return cv2.bitwise_and(frame, frame, mask=mask), mask

def remove_reflections(frame, threshold=240, mask=None):
    if mask is not None:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, reflection_mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
    reflection_mask = cv2.medianBlur(reflection_mask, 5)

    cleaned_frame = cv2.inpaint(frame, reflection_mask, 7, cv2.INPAINT_TELEA)
    return cleaned_frame

def calculate_patina_area(frame, combined_mask, ellipse_axes_main):
    """
    Calcola l'area relativa della patina, applicando la maschera combinata.
    Normalizza l'area in base all'area ellittica.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Fase di pre-processing
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = cv2.absdiff(gray, blurred)

    _, patina_mask = cv2.threshold(diff, 1.5, 255, cv2.THRESH_BINARY)

    # Applica la maschera combinata (ellisse principale + esclusioni)
    patina_mask = cv2.bitwise_and(patina_mask, combined_mask)

    patina_area = np.sum(patina_mask > 0)
    ellipse_area = np.pi * ellipse_axes_main[0] * ellipse_axes_main[1]  # Area ellittica

    relative_area = patina_area / ellipse_area if ellipse_area > 0 else 0
    return relative_area, patina_mask

def save_results(results, output_dir, file_name):
    # Salvataggio dei dati numerici in un file di testo
    txt_file = os.path.join(output_dir, f"{file_name}_patina_analysis.txt")
    with open(txt_file, 'w') as file:
        file.write("Tempo (s)\tArea Relativa (%)\n")
        for time, area in results:
            file.write(f"{time:.2f}\t{area * 100:.2f}\n")
    print(f"Risultati salvati in: {txt_file}")

    # Salvataggio del grafico
    img_file = os.path.join(output_dir, f"{file_name}_patina_graph.png")
    plt.plot([x[0] for x in results], [x[1] * 100 for x in results])
    plt.title('Area relativa della patina in funzione del tempo')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Area Relativa (%)')
    plt.grid(True)
    plt.savefig(img_file)
    print(f"Grafico salvato in: {img_file}")

# Iniziamo a processare il video
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()

if success:
    selected_frame, mask = manual_contour_selection(frame)
    previous_frame = frame.copy()

    frame_count = 0
    relative_areas = []
    results = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while success:
        # Step 1: Remove reflections
        frame_no_reflection = remove_reflections(frame, mask=mask)

        # Step 2: Calcolare l'area della patina
        relative_area, patina_mask = calculate_patina_area(frame_no_reflection, mask, ellipse_axes_main)
        relative_areas.append(relative_area)

        # Registrazione dei risultati
        time_elapsed = frame_count / fps
        results.append((time_elapsed, relative_area))
        print(f"Frame {frame_count}: Tempo: {time_elapsed:.2f}s - Area relativa della patina: {relative_area:.2%}")

        # Visualizzazione
        overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(patina_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.putText(overlay, f"Tempo: {time_elapsed:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Area Relativa: {relative_area:.2%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Patina Detection in Tempo Reale", overlay)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy()
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Salva i risultati e il grafico
    save_results(results, output_dir, file_name)
