import tkinter as tk
import torch

from torch import nn
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

from utils import define_device
from dataPreparing.loaders import ModelsLoader


class GUI:
    def __init__(self, model: nn.Module, device: str, preprocess):
        self.window = tk.Tk()
        self.window.title("Gender Classification")
        self.window.geometry("500x500")
        self.window.configure(bg="lightgray")

        self.model = model
        self.device = device
        self.preprocess = preprocess

        self.model.eval()

        self.label = tk.Label(self.window, text="Upload an image to classify", font=("Arial", 16), bg="lightgray")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self.window, text="Upload Image", command=self.upload_image,
                                       font=("Arial", 12), bg="lightgray")
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=20)

        self.result_label = tk.Label(self.window, text="", font=("Arial", 16), bg="lightgray")
        self.result_label.pack()

        self.window.mainloop()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                image = Image.open(file_path).convert("RGB")
                self.display_image(image)
                self.classify_image(image)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def display_image(self, image):
        img_resized = image.resize((224, 224))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def classify_image(self, image):
        x = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predict = self.model(x).item()

        self.display_result(predict)

    def display_result(self, predict):
        labels_map = {0: "Female", 1: "Male"}

        gender_class = round(predict)

        if gender_class == 0:
            prob = 1.0 - predict
        else:
            prob = predict

        result_text = f"{labels_map[gender_class]} with {prob:0.2f}"
        self.result_label.config(text=f"Prediction: {result_text}")


if __name__ == "__main__":
    device = define_device()

    models_loader = ModelsLoader(device)
    model = models_loader.load_local("./models/gc-simple-dnn-2.pth")
    _, preprocess = models_loader.load_vit_b_16()

    app = GUI(model, device, preprocess)
