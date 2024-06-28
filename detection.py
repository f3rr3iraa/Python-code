import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2

def load_reference_images(reference_image_paths):
    reference_images = []   
    for image_path in reference_image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0) 
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        reference_images.append(thresh)
    return reference_images

def detect_reference_objects(reference_images):
    reference_contours = []
    for reference_image in reference_images:
        contours, _ = cv2.findContours(reference_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reference_contours.append(contours)
    return reference_contours

def compare_areas(contour, reference_contours):
    min_diff = float('inf')
    best_match = None
    for i, reference_contours_list in enumerate(reference_contours):
        for reference_contour in reference_contours_list:
            reference_area = cv2.contourArea(reference_contour)
            area_diff = abs(cv2.contourArea(contour) - reference_area)
            if area_diff < min_diff:
                min_diff = area_diff
                best_match = i
    return best_match


def detect_all_objects(image_path, text_widget, reference_contours):
    if not image_path:
        return None, None
    
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (11, 11), 0) 
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5) 
    
    kernel = np.ones((7, 7), np.uint8)  
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 5  
    max_area = 5000 
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    fig, ax = plt.subplots()  
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    object_info = ""
    for i, contour in enumerate(filtered_contours):
        contour = np.squeeze(contour)
        polygon = Polygon(contour, edgecolor='aqua', linewidth=1, fill=None)  
        ax.add_patch(polygon)
        match_index = compare_areas(contour, reference_contours)
        if match_index is not None:
            centroid = np.mean(contour, axis=0)
            category_text = None
            category_num = None
            if match_index == 0:
                category_text = "Mosquito pequeno"
                category_num = "1"
            elif match_index == 1:
                category_text = "Mosquito grande"
                category_num = "2"
            elif match_index == 2:
                category_text = "Mosca"
                category_num = "3"

            if category_text:
                ax.text(centroid[0], centroid[1], category_num, color='blue', fontsize=8, ha='center', va='center')
            
            object_info += f"Nº do objeto {i+1}: {len(contour)} pixels - Categoria: {category_text}\n"
    
    ax.axis('on')
    
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, object_info)
    text_widget.config(state=tk.DISABLED)
    
    return len(filtered_contours), object_info, fig


def main():
    root = tk.Tk()
    root.title("Object Detection")
    
    def on_closing():
            exit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.config(yscrollcommand=scrollbar.set)
    
    text_widget = tk.Text(frame, wrap=tk.WORD, width=80, height=20)
    text_widget.pack(side=tk.RIGHT, fill=tk.Y)
    
    reference_image_paths = ["cat1.jpg", "cat2.jpg", "cat3.jpg"]
    reference_images = load_reference_images(reference_image_paths)
    
    reference_contours = detect_reference_objects(reference_images)
    
    def open_image():
        image_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if image_path:
            num_objects, object_info, fig = detect_all_objects(image_path, text_widget, reference_contours)
            if num_objects is not None:
                canvas.delete("all")
                canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
                canvas_widget.draw()
                canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                
                text_widget.config(state=tk.NORMAL)
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, object_info)
                text_widget.config(state=tk.DISABLED)
    
    open_button = tk.Button(root, text="Insert Image", command=open_image)
    open_button.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()