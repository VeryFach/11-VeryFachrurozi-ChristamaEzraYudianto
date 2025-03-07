import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Muat model dan scaler
with open('trained_model.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)

# Periksa apakah file menyimpan tuple (Scaler, Model)
if isinstance(loaded_objects, tuple):
    scaler, model = loaded_objects  # Ambil scaler dan model
else:
    model = loaded_objects
    scaler = None  # Jika scaler tidak disimpan dalam file, gunakan None

# Fungsi untuk melakukan prediksi
def test_model(model, data, scaler=None):
    if scaler is not None:
        data = scaler.transform(data)  # Normalisasi fitur sebelum prediksi
    return model.predict(data)

# Streamlit app
st.title("Dashboard Pengujian Model ML")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset ObesityDataSet.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diupload:")
    st.write(data)

    # Pastikan semua kolom yang dibutuhkan ada dalam dataset
    required_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Gender', 'NObeyesdad']
    if all(col in data.columns for col in required_columns):

        # Pisahkan fitur dan target
        X = data[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Gender']]
        y = data['NObeyesdad']

        # Lakukan prediksi
        predictions = test_model(model, X, scaler)

        # Tampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        st.write(predictions)

        # Hitung dan tampilkan akurasi
        accuracy = accuracy_score(y, predictions)
        st.write(f"Akurasi: {accuracy:.2f}")

        # Tampilkan classification report
        st.write("Classification Report:")
        st.text(classification_report(y, predictions))

    else:
        st.write("Error: Dataset tidak memiliki semua kolom yang dibutuhkan.")
