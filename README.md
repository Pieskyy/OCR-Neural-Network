# OCR---Neural-Network
OCR stands for Optical Character Recognition.

A Neural Network is a machine learning model inspired by the human brain.
it is designed to recognize patterns, process data, and solve complex problems

The Neural Network in this Repo is Designed to guess the number you drew.

How to use:
    Run the train_model.py (you only need to do this once and it SHOULD save the data)
    run the predicting.py (run whenever you wish to use)


Understanding this Repo:

#what-is-a-neural-network?
#the-nueron
#activation-functions
#nueral-network-layers
#ocr
#loss-functions
#backpropagation
#gradient-descent
#flattened-images

    
what-is-a-neural-network?

A neural network is a math model that learns patterns by adjusting numerical parameters called weights. It approximates a function f(x) ≈ y, where x is input data and y is the desired output

    A neuron computes a weighted sum of inputs and applies an activation function:
        z = w1x1 + w2x2 + ... + wnxn + b
        a = σ(z)

    Where w are weights, x are inputs, b is bias, and σ is an activation function.

Think of a neural network like a student learning to recognize animals:
    Show them a cat -> they say "cat" (if correct, good! if wrong, learn from mistake)
The network starts completely clueless but gets better with practice
It's basically a big math equation that adjusts itself until it gets things right
Simple analogy: Like tuning a guitar. you adjust strings until the note sounds right


