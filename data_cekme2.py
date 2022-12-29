import pandas as pd
import warnings
import itertools    
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input, Dropout, BatchNormalization, GRU
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

sns.set_style('dark')

warnings.filterwarnings('ignore')




from scipy.io import loadmat  




data= loadmat("transformed.mat")
transformed=data["transformed"]


columns = ["signal", "label"]
index = range(2000)

df = pd.DataFrame(index=index, columns = columns)
print(df)




for j in range(1000):
    df["label"][j] = 0
    df["signal"][j] = np.array(data["transformed"][0, : ,j])


for j in range(1000,2000):
    df["label"][j] = 1
    k = j-1000
    df["signal"][j] = np.array(data["transformed"][1, :, k])



print(df)

print(df.info())

print(df.head())

print(df.groupby("label").count())


print(f'Classes are almost balanced. We can get away with the difference.\n{df.label.value_counts()}')
df.label.value_counts().plot(kind='bar', color='tab:blue', title='Label')
plt.show()
    



t = range(len(df["signal"][0]))
# df["signal"][0].plot(title='0. signal', color='tab:blue', ax=axes[0])
plt.subplot(4,1,1)
plt.plot(t,df["signal"][0] )
plt.subplot(4,1,2)
plt.plot(t,df["signal"][999])
plt.subplot(4,1,3)
plt.plot(t,df["signal"][1000])
plt.subplot(4,1,4)
plt.plot(t,df["signal"][1499])
plt.show()



import seaborn as sns
sns.countplot(x= 'label', data=df)
plt.show()


# Split  data

Y = df['label'].copy()
X = df.drop('label', axis=1).copy()

print(Y.head())

print(X.head())


X_train, x_test, Y_train, y_test = train_test_split(X, Y,random_state=111, test_size=0.3)
X_train, x_val, Y_train, y_val=train_test_split(X_train, Y_train, random_state=111, test_size=0.3)

print(type(X_train))
print("******************")
print(X_train.shape)

X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))

x_test = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1],1))

print(type(X_train))


Y_train = pd.get_dummies(Y_train)
print(X_train.shape)
print(Y_train.shape)
print("######################")
y_test = pd.get_dummies(y_test)
y_val = pd.get_dummies(y_val)
Y_train = np.array(Y_train).reshape((Y_train.shape[0],Y_train.shape[1],1))

### 3 - Modeling 

# 3.1 - LSTM


i_lstm = Input(shape=(X_train.shape[1], 1))

x_lstm = LSTM(256, return_sequences=True)(i_lstm)
x_lstm = Flatten()(x_lstm)
y_lstm = Dense(3, activation='softmax')(x_lstm)

model_lstm = Model(i_lstm, y_lstm)

model_lstm.summary()




adam = Adam(learning_rate=0.001)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
mc = ModelCheckpoint('./best_lstm_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
                                    
model_lstm.compile(optimizer=adam,
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])


X_train = list(X_train)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)

Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

print(X_train.shape)
print(Y_train.shape)


lstm_h = model_lstm.fit(X_train, Y_train,
                   batch_size=32,
                   validation_data=(x_val, y_val),
                   epochs=50,
                   callbacks=[es, mc, lr_schedule])

        






