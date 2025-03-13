Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is a skip connection?
What is a bottleneck layer?
What is the Inception Network?
What is ResNet? ResNeXt? DenseNet?
How to replicate a network architecture by reading a journal article
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2) and tensorflow (version 2.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except from tensorflow import keras as K
All your files must be executable
The length of your files will be tested using wc

A convolutional neural network or CNN is a tool that is very usesful in training a model for computer vision and image recognition tasks. These would do a lot of the work in programs that depend on facial recogniton or in the case of computer vision, helping self driving cars identify obstacles and allow the cameras to perceive the world around them in a way that can then lead to a positive outcome. A Deep CNN is similar except for the fact that they leverage convolutional layers to learn hierarchical features, pooling layers to reduce dimensionality, and fully connected layers for high-level reasoning. Their ability to automatically extract relevant features and their robustness to spatial variations make them essential for a wide range of computer vision applications. It is like taking an image and breaking it up into a grid and then examing each tiny square in that grid one by one until an image emerges. There are a few different techniques that form a standard operating procedure depending on the task we are expected to perform. The requirements are questions to be answered are above.
