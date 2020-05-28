from keras.layers import Conv2D,Flatten,Dense,Dropout,MaxPooling2D
from keras.models import Sequential
from keras.datasets import mnist

#Loading Datasets of mnist for handwritten digits
from keras.optimizers import SGD
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=mnist.load_data()

img_rows=x_train[0].shape[0]
img_cols=x_train[1].shape[0]
#print(x_train.shape)

#adding 4th dimension using reshape
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

#shape of image in terms of rows,cols,number of colors (1(B/W) / 3(RGB))
input_shape=(img_rows,img_cols,1)

#conveting train and test data to float32 type
x_train=x_train.astype("float32")
x_test=x_test.astype("float32")

#normalized to 0 and 1 by dividing it 255 (Max Value)
x_train/=255
x_test/=255

print("Train Shape : ",x_train.shape[0])
print("Test Shape : ",x_test.shape[0])
print("Input Shape : ",input_shape)

#Hot One Encoding of labels both y_train and y_test
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("Number of Classes : ",y_test.shape[1])
num_classes=y_test.shape[1]
num_pixels=x_train.shape[1] * x_train.shape[2]

#creating Model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

#Compile Model
model.compile(loss='categorical_crossentropy',optimizer=SGD(0.01),metrics=['accuracy'])
print(model.summary())

#Train Model
batch_size=128
epochs=10
history=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

model.save("Digit_Recognition.h5")
print("Model Saved")