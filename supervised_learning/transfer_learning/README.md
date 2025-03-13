General
What is a transfer learning?
What is fine-tuning?
What is a frozen layer? How and why do you freeze a layer?
How to use transfer learning with Keras applications
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

Transfer learning is the process of utilizing a model that has been trained on data of a similar nature to your data. You can use parts of that model that has already been trained on your data set. For example, if you are trying to train a model on the CIFAR-10 dataset which is an image classification data set, transfer learning could lead you to using a model that was trained on ImageNet which is also an image classification data set. It is not an exact science and requires fine tuning of parameters to improve the validation accuracy and reduce over and under fitting. When using the model you will freeze certain layers meaning they cannot be changed on subsequent iterations or further epochs. This allows you to determine just how much of the previously trained model you want to use. This is the most "cost-effective" way to train models and because of the scarcity of large data sets, transfer learning is becoming even mroe useful and valuable as time goes on.
