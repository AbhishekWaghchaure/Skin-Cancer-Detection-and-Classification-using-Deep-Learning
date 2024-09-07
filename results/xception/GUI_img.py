import numpy as np
import cv2
from tkinter import*
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
# import joblib
# import json
import keras
from datetime import date
from datetime import datetime
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import DenseNet201
from matplotlib import pyplot as plt
import time
import keras
from keras.optimizers import Adam
from keras.applications import Xception
from keras.preprocessing import image

parent=Tk()
parent.geometry("700x400")
parent.resizable(0,0) 
parent.title("Application")
parent.configure(background="Forest green")
filename=""

#===========model for CNN===================
img_width, img_height = 71, 71

input_shape = (img_width, img_height, 3)

# Use Xception as the base model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss ='categorical_crossentropy',optimizer=opt, metrics =['accuracy'])


model.load_weights('model_xcpetion.h5')
print("done loading weights of Xception trained model")

labels = ['akiec','bcc','bkl','df','mel','nv','vasc']

# def input1():
#     global filename
#     global panel1
#     filename = askopenfilename()
#     img = cv2.imread(filename)
#     img1= cv2.resize(img,(256,256))
    
#     # preprocess for normal-abnormal
#     cropped = cv2.resize(img,(71,71))
#     cv2.imwrite( "cropped.jpg",cropped)
#     cropped= cropped/255
#     cropped = np.reshape(cropped,[1,71,71,3])
#     img3 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)   
#     cv2.waitKey(10)
       
#     img2 = Image.fromarray(img3) 
#     img1 = img2.resize((200, 200), Image.LANCZOS)  
#     img1 = ImageTk.PhotoImage(img1)
#     panel1=Label(parent ,image=img1)
#     panel1.image = img1
#     panel1.place(x=30, y=120)    
#     panel1.config() 
def input1():
    global filename
    global panel1
    filename = askopenfilename()
    
    if filename:  # Check if a file was selected
        try:
            img = cv2.imread(filename)
            if img is not None:  # Check if the image was read successfully
                img1 = cv2.resize(img, (256, 256))
                cropped = cv2.resize(img, (71, 71))
                cv2.imwrite("cropped.jpg", cropped)
                cropped = cropped / 255
                cropped = np.reshape(cropped, [1, 71, 71, 3])
                img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                cv2.waitKey(10)

                img2 = Image.fromarray(img3)
                img1 = img2.resize((200, 200), Image.LANCZOS)
                img1 = ImageTk.PhotoImage(img1)
                panel1 = Label(parent, image=img1)
                panel1.image = img1
                panel1.place(x=30, y=120)
                panel1.config()
            else:
                # Display error message if the image couldn't be read
                messagebox.showerror("Error", "Failed to read the image file.")
        except Exception as e:
            # Handle any other exceptions that might occur
            messagebox.showerror("Error", str(e))
    
def recognition():
    global count
    global panel3
    global filename
    global features
    global img
    global img1
    global panel1
    
    print(filename)
    
    sub_img1 = cv2.imread('cropped.jpg')
    sub_img = image.load_img('cropped.jpg', target_size=(img_width, img_height))
    
    x = image.img_to_array(sub_img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    
    # Make prediction
    prediction = model.predict(x)
    
    output = np.argmax(prediction)
    print(labels[output])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sub_img1, labels[output], (15,15), font, 0.25, (0, 255, 0), 1, cv2.LINE_AA)           
    cv2.waitKey(10)   
    abcd = labels[output]
    Output1=Label(parent,bg='white', text=abcd).place(x=470, y=330)     
    panel1.config() 
    
    sub_img1 = cv2.cvtColor(sub_img1,cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(sub_img1) 
    img1 = img2.resize((200, 200), Image.LANCZOS)  
    img1 = ImageTk.PhotoImage(img1)
    panel1=Label(parent ,image=img1)
    panel1.image = img1
    panel1.place(x=370, y=120)
    

def main(): 
    image1 = Image.open("bg1.jpg")
    test = ImageTk.PhotoImage(image1)
    label1 = Label(image=test)
    label1.image = test
    label1.place(x=0, y=0)
    heading = Label(parent, bg='orange', text="     Skin Cancer Recognition     ",font=("Courier", 26)).place(x=0, y=0)  
    btn1=Button(parent,text="Input Image",bg='firebrick1', activebackground='red', height=1,width=10,command=input1).place(x=100, y=90)
    btn2=Button(parent,text="Recognition",bg='firebrick1', activebackground='red', height=1,width=10,command=recognition).place(x=420, y=90)
    # outlable = Label(parent, bg='red', text=" Output",font=("Courier", 10)).place(x=600, y=90)
 
    parent.mainloop()

if __name__ == "__main__":      
    main()