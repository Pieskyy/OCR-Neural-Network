OCR Neural Network AI

This project is essentially a home made AI that guesses numbers.


What is OCR?
OCR stands for Optical Character Recognition. it's technology that converts images of text into machine-readable characters. In this project, i using OCR to recognize handwritten digits (0-9)

Think of it like teaching a computer to read your handwriting



What is a Neural Network?
A Neural Network is a machine learning model inspired by how the human brain works



How it works:
You show it a bunch of handwritten numbers

It makes guesses and learns from its mistakes

Over time, it gets better and better at recognizing digits

Eventually, it can guess numbers it's never seen before!



What Does This Project Do?
This project lets you draw a number on your screen, and the neural network will try to guess what you drew



Files in This Project:
File	What it does
train_model.py	Trains the neural network on thousands of handwritten digits (only need to run once)
predicting.py	Launches the drawing app where you can test the model


How to Use This Project:

Step 1: Install the Requirements:

    torch

    torchvision

    pillow 

    numpy

    tkinter


Step 2: Train the Model (One Time Only):
    
    run train_model.py

This will:

Download the MNIST dataset (60,000 handwritten digits)

Train our neural network for 5 rounds (epochs)

Save the trained model as ocr_model.pth

What you should see:

text
Epoch 1/5, Loss: 0.4321
Epoch 2/5, Loss: 0.2345
Epoch 3/5, Loss: 0.1876
Epoch 4/5, Loss: 0.1543
Epoch 5/5, Loss: 0.1321
Model saved as ocr_model.pth

(The numbers will be different but should get smaller each time (which means its learning))


Step 3: Test It Out

    run predicting.py

This opens a drawing window where you can:

    Draw a number with your mouse

    Click "Guess" to see what the AI thinks

    Click "Clear" to try again



How Does the Neural Network Work?

The Architecture:
    the network has 4 layers:

        Input Layer: Sees 784 pixels (28×28 image flattened into one line)

        Hidden Layer 1: 128 neurons that look for simple patterns (edges, curves)

        Hidden Layer 2: 64 neurons that combine patterns into shapes

        Output Layer: 10 neurons (one for each digit 0-9)

    The Learning Process
        Forward Pass: The image flows through the network

        Prediction: The network guesses a number

        Loss Calculation: We measure how wrong it was

        Backward Pass: The network figures out which neurons caused the mistake

        Weight Update: All neurons adjust slightly to do better next time

    Repeat thousands of times and gets smarter


sources:
    https://www.youtube.com/watch?v=2S1dgHpqCdk

    https://codemy.com/

    https://youtu.be/52NXldtvOnE?si=L3A4A54q4nQbTNhx

    i used some other videos but forgot to grab the link.



IF YOU HAVE ANY QUESTIONS/CONCERNS FEEL FREE TO ASK
