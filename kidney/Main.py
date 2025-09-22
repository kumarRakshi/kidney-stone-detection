import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
from tkinter import *
import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import argparse
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import glob
import math
import time
import os
import itertools
#import requests
from PIL import Image
from numpy import average, linalg, dot

import matplotlib.pyplot as plt
import numpy as np
import argparse

#from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from PIL import Image, ImageStat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew

import math
import argparse


import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import math

import glob
from keras.models import Sequential, load_model
#import preprocess as pre
import train as tr
import Predict_CNN as pred
import SVM_train as svmtrain
import Predict_SVM as predsvm

bgcolor="#87CEEB"
bgcolor1="#87CEEB"
fgcolor="black"
def Home():
        global window
        def clear():
            print("Clear1")
               
            txt1.delete(0, 'end')    
            
            txt3.delete(0, 'end')    



        window = tk.Tk()
        window.title("Kidney Stone Detection Using Deep Learning")

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)
        bg = PhotoImage(file = "4.png")
        # Show image using label 
        label1 = Label( window, image = bg) 
        label1.place(x = 0, y = 0) 

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Kidney Stone Detection Using Deep Learning" ,bg=bgcolor  ,fg=fgcolor  ,width=100  ,height=3,font=('times', 20, 'italic bold underline')) 
        message1.place(x=10, y=20)

        

        lbl1 = tk.Label(window, text="Select  Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=270)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=275)

        

        lbl3 = tk.Label(window, text="Select Image",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl3.place(x=100, y=340)
        
        txt3 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt3.place(x=400, y=345)
        

        def browse():
                path=filedialog.askdirectory()
                print(path)
                txt1.delete(0, 'end')
                txt1.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select DataSet Folder")     
              

        def browse3():
                path=filedialog.askopenfilename()
                print(path)
                txt3.delete(0, 'end')
                txt3.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Input Image")        

        
        def trainingcnnmodel():
                sym=txt1.get()
                if sym !="":
                        tr.process(sym)
                        tm.showinfo("Output", "CNN Training Completed Successfully")
                else:
                        
                        tm.showinfo("Input error", "Select DataSet Folder")
        def trainingsvmmodel():
                sym=txt1.get()
                if sym !="":
                        svmtrain.process()
                        tm.showinfo("Output", "SVM Training Completed Successfully")
                else:
                        
                        tm.showinfo("Input error", "Select DataSet Folder")
        def predictingcnnresult():
                sym=txt3.get()
                if "CT_images" in sym:
                        if sym !="":
                                res=pred.predict(sym)
                                tm.showinfo("Output", "Predicted as "+str(res))
                        else:
                                tm.showinfo("Input error", "Select Image")
                else:
                        tm.showwarning("Invalid Input","Select X-Ray Or CT image")
        def predictingsvmresult():
                sym=txt3.get()
                if sym !="":
                        if "CT_images" in sym:
                                res=predsvm.process(sym)
                                tm.showinfo("Output", "Predicted as "+str(res))
                        else:
                                tm.showwarning("Invalid Input","Select X-Ray Or CT image")

                else:
                        
                        tm.showinfo("Input error", "Select Image")



 

        
        browse1btn = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse1btn.place(x=650, y=270)

        

        browse3btn = tk.Button(window, text="Browse", command=browse3  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse3btn.place(x=650, y=340)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=950, y=200)
         
        

        


        PRbutton = tk.Button(window, text="CNN Training", command=trainingcnnmodel  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        PRbutton.place(x=100, y=500)
        PRbutton1 = tk.Button(window, text="SVM Training", command=trainingsvmmodel  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        PRbutton1.place(x=300, y=500)

        DCbutton = tk.Button(window, text="CNN Prediction", command=predictingcnnresult  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        DCbutton.place(x=500, y=500)
        DCbutton1 = tk.Button(window, text="SVM Prediction", command=predictingsvmresult  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        DCbutton1.place(x=700, y=500)



        quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=900, y=500)

        window.mainloop()
Home()

