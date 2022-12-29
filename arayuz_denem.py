import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import keras


model = keras.models.load_model('best_lstm_model.h5')

classes = { 
    0:'its a Yes',
    1:'its a No',
}


# Initialise GUI

top = tk.Tk()
top.geometry("800x600")
top.title("Yes-No Binary Classification")
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

