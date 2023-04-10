import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Label # Import Label from ttk module
import tensorflow as tf
from PIL import ImageTk
import numpy as np
window = tk.Tk()
window.title("Dog Breed Classifier")
window.geometry("720x360")
window.config(bg="#F0F0F0")

def upload_image():
    filename = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.png")])
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.expand_dims(image, axis=0)
    return image

def run_model():
    image = upload_image()
    model = tf.keras.models.load_model("C:\\Users\\harsh\\OneDrive\\Documents\\GitHub\\dog_breed_classification\\Dog_Breed_Classification\\models\\dog_v1")
    prediction = model.predict(image)
    class_index = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = round(100 * (np.max(prediction[0])), 2)
    labels = ['French Bulldog','German Shepherd','Golden Retriever','Poodle','Yorkshire Terrier']
    result = f"The image is a {labels[class_index]} with confidence {confidence}%."
    print(result)
    # Use the label object to display the result
    result_label.config(text=result) 
    pil_image = tf.keras.preprocessing.image.array_to_img(image[0])
    tk_image = ImageTk.PhotoImage(pil_image)
    img_label.config(image=tk_image)
    img_label.image = tk_image

button = tk.Button(window, text="Upload and classify an image", command=run_model)
button.pack(pady=10)

canvas = tk.Canvas(window, width=600, height=100, bg="#f0f0f0")
canvas.pack(pady=10)

result_label = Label(canvas, text="", style="Result.TLabel",background="#f0f0f0", font=("Helvetica", 12), anchor="center")
result_label.place(relx=0.5, rely=0.5, anchor="center")

img_label = tk.Label(window,width=256, height=256)
img_label.pack(pady=10)
window.mainloop()