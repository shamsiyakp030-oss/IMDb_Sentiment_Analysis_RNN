#  IMDb Sentiment Analysis using RNN

> Professional GitHub README for an IMDb Sentiment Analysis project with
> a Streamlit app.

##  Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)

------------------------------------------------------------------------

#  Overview

This project implements an end-to-end Natural Language Processing (NLP)
solution that classifies IMDb movie reviews as **Positive** or
**Negative** using a **Simple Recurrent Neural Network (SimpleRNN)**. It
also includes a modern **Streamlit web application** for real-time
sentiment prediction.

##  Features

-   IMDb sentiment classification
-   Text preprocessing & tokenization
-   Embedding + SimpleRNN model
-   Real-time Streamlit prediction
-   Confidence score
-   Modern UI
-   Model saving/loading
-   Accuracy & loss visualization

##  Screenshots

Place screenshots in `assets/`.

``` text
assets/
├── home.png
├── streamlit-home.png
├── streamlit-result.png
├── accuracy.png
├── loss.png
```

##  Streamlit App

Run:

``` bash
pip install -r requirements.txt
streamlit run review.py
```

Open: `http://localhost:8501`

##  Workflow

``` text
IMDb Dataset
   ↓
Cleaning
   ↓
Tokenization
   ↓
Padding
   ↓
Embedding
   ↓
SimpleRNN
   ↓
Prediction
```

##  Project Structure

``` text
IMDb-Sentiment-Analysis-RNN/
├── review.py
├── IMDb_Sentiment_Analysis_RNN.ipynb
├── sentiment_rnn_model.keras
├── word_index.pkl
├── requirements.txt
├── assets/
└── README.md
```

##  Installation

``` bash
git clone https://github.com/yourusername/IMDb-Sentiment-Analysis-RNN.git
cd IMDb-Sentiment-Analysis-RNN
python -m venv venv
pip install -r requirements.txt
```

##  Technology Stack

-   Python
-   TensorFlow
-   Keras
-   Streamlit
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
## streamlit cloud community
- https://imdbsentimentanalysisrnn-hp6dsyycy6wprpb7nfarqe.streamlit.app/

##  Results

-   Binary sentiment classification
-   Accuracy/Loss visualization
-   Real-time inference

##  Future Improvements

-   LSTM/GRU
-   Attention
-   Transformer models
-   Docker
-   Cloud deployment

##  Author

**Shamsiya KP**

##  Support

If you like this project, please give it a ⭐ on GitHub.

