Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is OpenCV and how do you use it?
What is object detection?
What is the Sliding Windows algorithm?
What is a single-shot detector?
What is the YOLO algorithm?
What is IOU and how do you calculate it?
What is non-max suppression?
What are anchor boxes?
What is mAP and how do you calculate it?
Requirements
Python Scripts
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
All your files must be executable
The length of your files will be tested using wc

This was the final project of the third trimester and represent the culmination of a deep dive into computer vision and all of the processes that are utilized in order to train computer vision models. We started with using computer vision to identify hand drawn digits beteween 0-9 and in this final project, the model would take a photograph and be capable of identifying one of 24 objects with a relatively high degree of accuracy anywhere in the picture. It was able to correctly identify objects that were partially obscured or cropped and do so with multiple objects in one photograph. Object Detection is the method that self driving cars use as their form of computer vision. It helps them identify a pedestrian, another vehicle or any other objects that might be found in close proximity to a road way. It determines the location of the object as well as its shape determining from a preset list of labels what it actually is.
