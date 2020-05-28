from keras.models import load_model

classifier=load_model("Digit_Recognition.h5")
print("Model Loaded")