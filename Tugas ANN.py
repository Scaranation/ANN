import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data input dan output
x1 = np.array([5, 3, 9, 4])
x2 = np.array([2, 8, 1, 4])
y = np.array([19, 11, 23, 8])

# Gabungkan x1 dan x2 menjadi satu array input dengan shape (4, 2)
X = np.column_stack((x1, x2))

# Custom callback to print loss and weights at each epoch
# class PrintWeightsAndLoss(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f'Epoch {epoch + 1}: Loss = {logs["loss"]}')
#         for layer in self.model.layers:
#             weights, biases = layer.get_weights()
#             print(f'Layer: {layer.name} Weights: {weights} Biases: {biases}')

# Bangun model ANN
model = Sequential()
model.add(Dense(100, input_dim=2, activation='relu', name='hidden_layer'))  # Layer input dengan 2 neuron dan hidden layer dengan 10 neuron
model.add(Dense(1, activation='linear', name='output_layer'))  # Layer output dengan 1 neuron

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# Latih model 
model.fit(X, y, epochs=100)

# Prediksi nilai y untuk x1=2, x2=2
x1_new = 2
x2_new = 2
X_new = np.array([[x1_new, x2_new]])
y_pred = model.predict(X_new)

print(f'\nPrediksi y untuk x1={x1_new} dan x2={x2_new} adalah {y_pred}')
