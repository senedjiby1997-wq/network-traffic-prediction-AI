import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 📁 Définir les chemins dynamiquement
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'network_traffic_sample.csv')
results_dir = os.path.join(base_dir, '..', 'results')

# 📂 Créer le dossier results s’il n’existe pas
os.makedirs(results_dir, exist_ok=True)

# 📊 Charger les données
data = pd.read_csv(data_path)
values = data['traffic'].values.reshape(-1, 1)

# 🔄 Normalisation
scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

# 🧩 Préparation des séquences temporelles
X, y = [], []
time_steps = 10
for i in range(len(values_scaled) - time_steps):
    X.append(values_scaled[i:i+time_steps])
    y.append(values_scaled[i+time_steps])
X, y = np.array(X), np.array(y)

# 🧠 Création du modèle LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 🚀 Entraînement du modèle
model.fit(X, y, epochs=30, batch_size=16, verbose=1)

# 🔮 Prédiction
predictions = model.predict(X)
predicted = scaler.inverse_transform(predictions)
real = scaler.inverse_transform(y.reshape(-1, 1))

# 📈 Visualisation et sauvegarde du graphique
plt.figure(figsize=(10,5))
plt.plot(real, label='Réel')
plt.plot(predicted, label='Prédit', linestyle='--')
plt.legend()
plt.title('Prédiction du trafic réseau (LSTM)')
plt.xlabel('Temps')
plt.ylabel('Volume de trafic')
plt.grid(True)

# Enregistrer le graphique
plot_path = os.path.join(results_dir, 'traffic_plot.png')
plt.savefig(plot_path)
plt.show()

print(f"\n✅ Graphique enregistré dans : {plot_path}")


