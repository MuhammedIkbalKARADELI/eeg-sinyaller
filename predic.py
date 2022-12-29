import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras


sns.set_style('dark')

warnings.filterwarnings('ignore')

from scipy.io import loadmat

data= loadmat("z.mat")
#data = np.array(data)


print(type(data))
#print(data["z"])

data = data["z"]




raw = data[0]

transformed = data[1]

print(type(transformed))
print(transformed.shape)


plt.subplot(2,1,1)
plt.plot(raw)
plt.subplot(2,1,2)
plt.plot(transformed)
plt.show()


x_de = transformed.reshape(1,transformed.shape[0])
print(x_de)
print(type(x_de))
print(x_de.shape)


# # # Load the saved model
model = keras.models.load_model('best_lstm_model_new.h5')
# # # Use the model to make predictions
predictions = model.predict(x_de)

print(predictions)



if predictions[0][0] > predictions[0][1]:
    print("This is Yes ")
elif predictions[0][0] < predictions[0][1]:
    print("This is No ")






















