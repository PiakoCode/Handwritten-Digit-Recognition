import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
from ResNet18 import ResNet, Block

device = device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class HandwritingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwriting Digit Recognition")
        self.canvas = tk.Canvas(master, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.button_predict = tk.Button(master, text='Predict', command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(master, text='Clear', command=self.clear)
        self.button_clear.pack()
        self.label_result = tk.Label(master, text='', font=('Helvetica', 24))
        self.label_result.pack()
        self.image1 = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)
        self.model = torch.load("model.pth", map_location=device)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")
        self.label_result.config(text='')

    def predict(self):
        self.image1.save("temp_digit.png")
        image = Image.open("temp_digit.png").convert('L')
        image = image.resize((28, 28), Image.LANCZOS)
        
        image = np.array(image)
        image = 255 -image
        image = image.reshape(1, 1, 28, 28).astype('float32') / 255
        #===== model predict =====#
        image =  torch.tensor(image).to(device)
        self.model.eval()
        # print(image.shape)

        with torch.no_grad():
            out =  self.model(image)
            predict_digit = out.argmax(1)
        
        #===== model predict =====#
        # print(f'Predicted Digit: {predict_digit.item()}')
        self.label_result.config(text=f'Result: {predict_digit.item()}')
        


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()
