import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.special import boxcox1p
import requests
import io
import pickle

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to load the model and transformation parameters


def load_model_and_params(url_lambda, url_model):
    response_lambda = requests.get(url_lambda)
    response_model = requests.get(url_model)

    if response_lambda.status_code == 200 and response_model.status_code == 200:
        lambda_buffer = io.BytesIO(response_lambda.content)
        model_buffer = io.BytesIO(response_model.content)

        optimal_lambdas = pickle.load(lambda_buffer)

        model_file_path = 'Model/model.json'
        with open(model_file_path, 'wb') as file:
            file.write(model_buffer.getvalue())

        model = XGBRegressor()
        model.load_model(model_file_path)
        return model, optimal_lambdas
    else:
        st.error(
            "Failed to load model or transformation parameters. Please check the URLs.")
        return None, None


# Define all expected features
expected_features = ['POV', 'FOOD', 'ELEC',
                     'WATER', 'LIFE', 'HEALTH', 'SCHOOL', 'STUNTING']

# Home page
if st.session_state.page == 'home':
    st.title('Selamat Datang di PEKA: Pemantauan Ketahanan Pangan')

    # Custom CSS for text justification
    st.markdown(
        """
        <style>
        .justified-text {
            text-align: justify;
        }
        .highlight {
            font-weight: bold;
            color: #0056b3;  /* Optional: Change color to make it more visually appealing */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Using HTML within st.markdown to justify text and apply custom CSS classes
    st.markdown(
        """
        <div class="justified-text">
        Ketahanan pangan adalah aspek fundamental untuk mencapai kesejahteraan masyarakat yang berkelanjutan di Indonesia, 
        yang menghadapi tantangan signifikan karena berbagai faktor sosial-ekonomi seperti kemiskinan, akses pangan, 
        ketersediaan listrik dan air bersih, pendidikan, dan kesehatan. Penggunaan teknologi kecerdasan buatan (AI) dan 
        machine learning dapat memantau dan memprediksi faktor-faktor ketahanan pangan dengan model prediktif yang menganalisis 
        indikator-indikator seperti rasio konsumsi, kemiskinan, pengeluaran rumah tangga, akses listrik dan air bersih, harapan hidup, 
        tenaga kesehatan, pendidikan, dan prevalensi stunting. Salah satu inovasi yang dapat dikembangkan adalah sistem pemantauan 
        ketahanan pangan berbasis AI bernama PEKA, yang mengintegrasikan data dari berbagai sumber untuk memberikan informasi akurat 
        dan terkini tentang status ketahanan pangan di setiap wilayah.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("https://raw.githubusercontent.com/juanvalno/SEC/58ffc809a12294a4f73fbe334dfaa73de18f911b/Asset/Top%2010%20IKP.png")
    st.markdown(
        """
        <div class="justified-text">
        <span class="highlight">1. Top 10 Wilayah dengan Indeks Ketahanan Pangan Terendah</span><br>
        Pada grafik pertama, kita melihat daftar sepuluh wilayah dengan indeks ketahanan pangan (IKP) terendah. Wilayah Pegunungan 
        Bintang memiliki IKP terendah di antara wilayah-wilayah lain, yaitu 14,54, diikuti oleh Mamberamo Raya dengan 17,63. Sementara 
        itu, wilayah dengan IKP terendah lainnya adalah Dogiyai, Intan Jaya, dan Nduga.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="justified-text">
        Grafik ini mengindikasikan bahwa wilayah-wilayah dengan IKP rendah ini kemungkinan besar menghadapi tantangan signifikan dalam 
        hal ketersediaan dan aksesibilitas pangan. Ketahanan pangan yang rendah dapat disebabkan oleh berbagai faktor, termasuk 
        tingginya tingkat kemiskinan, infrastruktur yang buruk, serta akses yang terbatas terhadap pendidikan dan layanan kesehatan.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("https://raw.githubusercontent.com/juanvalno/SEC/58ffc809a12294a4f73fbe334dfaa73de18f911b/Asset/Bot%2010%20IKP.png")
    st.markdown(
        """
        <div class="justified-text">
        <span class="highlight">2. Perbandingan Persentase Kemiskinan (POV) dan Indeks Ketahanan Pangan (IKP) di 10 Wilayah dengan Kemiskinan Tertinggi</span><br>
        Grafik kedua menunjukkan hubungan antara persentase kemiskinan (POV) dan IKP di sepuluh wilayah dengan tingkat kemiskinan tertinggi. 
        Ada korelasi yang jelas antara kemiskinan dan ketahanan pangan; wilayah dengan tingkat kemiskinan tinggi cenderung memiliki IKP yang 
        rendah. Contohnya, wilayah Puncak Jaya memiliki persentase kemiskinan tertinggi, dan juga IKP yang sangat rendah.<br>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="justified-text">
        Fakta ini menegaskan bahwa peningkatan kondisi ekonomi adalah kunci untuk memperbaiki ketahanan pangan. Program pengentasan kemiskinan 
        yang efektif harus menjadi prioritas untuk meningkatkan aksesibilitas dan ketersediaan pangan di wilayah-wilayah ini.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("https://raw.githubusercontent.com/juanvalno/SEC/58ffc809a12294a4f73fbe334dfaa73de18f911b/Asset/Perbandingan%20Persentase%20Kemiskinan%20(POV)%20dan%20IKP.png")
    st.markdown(
        """
        <div class="justified-text">
        <span class="highlight">3. Distribusi Ketahanan Pangan Berdasarkan Cluster</span><br>
        Grafik ketiga menunjukkan distribusi IKP berdasarkan tiga cluster kondisi sosio-ekonomi:<br>
        <strong>Cluster 0:</strong> Klaster ini mencakup daerah-daerah dengan tingkat kemiskinan relatif rendah. Rata-rata IKP di klaster ini adalah 79,65, yang menunjukkan bahwa daerah-daerah ini memiliki ketahanan pangan yang relatif baik. Tingginya rata-rata lama sekolah perempuan dan ketersediaan layanan kesehatan yang cukup baik berkontribusi pada kondisi ini.<br><br>
        <strong>Cluster 1:</strong> Klaster ini terdiri dari daerah-daerah dengan tingkat kemiskinan sedang. Rata-rata IKP di klaster ini adalah 74,26. Meskipun sebagian besar rumah tangga memiliki akses listrik, masih ada banyak yang tidak memiliki akses air bersih, dan layanan kesehatan kurang memadai, menunjukkan perlunya peningkatan akses pendidikan dan kesehatan.<br><br>
        <strong>Cluster 2:</strong> Klaster ini mencakup daerah-daerah dengan tingkat kemiskinan sangat tinggi. Rata-rata IKP di klaster ini adalah yang terendah, yaitu 40,48. Klaster ini menghadapi tantangan terbesar, dengan akses pendidikan dan layanan kesehatan yang sangat terbatas, serta sebagian besar rumah tangga tidak memiliki akses listrik dan air bersih.    
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image("https://raw.githubusercontent.com/juanvalno/SEC/58ffc809a12294a4f73fbe334dfaa73de18f911b/Asset/Distribusi%20Ketahanan%20Pangan%20Berdasarkan%20Cluster.png")
    st.markdown(
        """
        <div class="justified-text">
        <span class="highlight">Analisis dan Implementasi Melalui Smart Monitoring System PEKA</span><br>
        Berdasarkan analisis data di atas, dapat disimpulkan bahwa ketahanan pangan di Indonesia sangat bervariasi tergantung pada kondisi 
        sosio-ekonomi masing-masing wilayah. Untuk mewujudkan ketahanan pangan yang lebih baik, perlu adanya pendekatan yang berbasis data 
        dan kecerdasan buatan melalui Smart Monitoring System PEKA (Pemantauan Ketahanan Pangan).<br><br>
        
        Sistem PEKA yang berbasis kecerdasan buatan (AI) dapat memainkan peran penting dalam memantau dan memprediksi ketahanan pangan, 
        serta memberikan informasi yang akurat dan terkini untuk pengambilan keputusan yang lebih baik. Implementasi sistem ini diharapkan 
        dapat membantu meningkatkan ketahanan pangan secara menyeluruh, terutama di wilayah-wilayah yang saat ini masih menghadapi tantangan
        besar.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button('Go to Prediction Page'):
        st.session_state.page = 'predict'

# Prediction page
elif st.session_state.page == 'predict':
    # Load the model and parameters
    model, optimal_lambdas = load_model_and_params(
        'https://raw.githubusercontent.com/juanvalno/SEC/6d0553bca78ed9b7479eb6f103ebcb1c2dca79b0/Model/transformation_params.pkl',
        'https://raw.githubusercontent.com/juanvalno/SEC/22d581a130216da15bff6c439d5cd7819258332d/Model/model.json'
    )

    if model is not None and optimal_lambdas is not None:
        st.title('Prediction Page')

        # Input fields
        input_data = {}
        col1, col2 = st.columns(2)
        for i, feature in enumerate(expected_features):
            if i < 4:
                input_data[feature] = col1.number_input(feature, value=0.0)
            else:
                input_data[feature] = col2.number_input(feature, value=0.0)

        # Create DataFrame with input data
        input_df = pd.DataFrame([input_data])

        # Ensure all inputs are numeric
        input_df = input_df.apply(pd.to_numeric)

        # Transform the inputs
        for feature, optimal_lambda in optimal_lambdas.items():
            if feature in input_df:
                input_df[feature] = boxcox1p(input_df[feature], optimal_lambda)

        # Reorder the columns to match the expected feature order
        input_df = input_df[expected_features]

        # Button to make predictions
        if st.button('Predict'):
            prediction = model.predict(input_df)
            inverse_prediction = np.expm1(prediction)
            st.write('Predicted IKP: {:.2f}'.format(inverse_prediction[0]))

        # Button to go back to the home page
        if st.button('Back to Home Page'):
            st.session_state.page = 'home'
