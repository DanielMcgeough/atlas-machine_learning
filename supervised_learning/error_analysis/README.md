Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

General
What is the confusion matrix?
What is type I error? type II?
What is sensitivity? specificity? precision? recall?
What is an F1 score?
What is bias? variance?
What is irreducible error?
What is Bayes error?
How can you approximate Bayes error?
How to calculate bias and variance
How to create a confusion matrix
Requirements
General
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9)
Your files will be executed with numpy (version 1.25.2)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should use the pycodestyle style (version 2.11.1)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module except import numpy as np
All your files must be executable
The length of your files will be tested using wc

The purpose of this project was to understand error analysis. Imagine you're training a dog to fetch a ball. You throw the ball, and sometimes the dog brings it back perfectly, but other times it brings back a stick, or runs off chasing a squirrel. Error analysis is like carefully examining those mistakes to understand why they happened. In machine learning, error analysis is the process of looking at the mistakes your model makes to figure out why it's making them. It's like a detective trying to solve a mystery, but instead of clues, you're looking at the data and the model's predictions. Certain types of errors can indicated problems with the data like overfitting or underfitting so understanding type of error you are receiving can help solve the issue. For instance if your model has a high accuracy rate on the training date but a low validation accuracy score your data set has fallen prey to overfitting meaning than the data is not generalized enough for pattern recognition. Once you identify the problem you can go about using techniques to remedy the problem.
