import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import numpy as np


class MyDigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Load the trained model
my_model = MyDigitClassifier()
my_model.load_state_dict(torch.load("ocr_model.pth"))
my_model.eval()  # Put in evaluation mode

# the drawing canvas
CANVAS_SIZE = 280  # EDIT SIZE HERE (280 is 10x the MNIST size, so we can draw bigger and then shrink it down)


window = tk.Tk()
window.title("Draw a Number (0-9)")
drawing_area = tk.Canvas(window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
drawing_area.pack()

# blank canvas
blank_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
pen = ImageDraw.Draw(blank_image)

# This function runs when you click and drag on the canvas
def draw_number(event):
    # Draw a circle where the mouse is (makes a thick line)
    x1 = event.x - 10
    y1 = event.y - 10
    x2 = event.x + 10
    y2 = event.y + 10

    drawing_area.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    pen.ellipse([x1, y1, x2, y2], fill=255)

# Bind mouse movement to drawing function
drawing_area.bind("<B1-Motion>", draw_number)

# recognize what number you drew
def guess_number():
    # Resize to 28x28 (MNIST size)
    small_image = blank_image.resize((28, 28))
    image_array = np.array(small_image) # Convert to numpy array

    image_array = image_array / 255.0  # Scale to 0-1
    image_array = (image_array - 0.5) / 0.5  # Scale to -1 to 1
    
    image_tensor = torch.tensor(image_array, dtype=torch.float32)# Convert to tensor (PyTorch format)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    output = my_model(image_tensor) # Get models guess (output is a list of probabilities for each digit)
    predicted_digit = torch.argmax(output).item() # Get the highest probability number
    
    result_display.config(text=f"{predicted_digit}")# Show the result

def clear_canvas():
    drawing_area.delete("all")
    pen.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    result_display.config(text="Draw")


predict_button = tk.Button(window, text="Guess", command=guess_number)
predict_button.pack()

clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

result_display = tk.Label(window, text="Draw", font=("Arial", 16))
result_display.pack()

window.mainloop()