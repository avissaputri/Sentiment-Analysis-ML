#import library
from flask import Flask, jsonify, request, make_response
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#inisiasi flask
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title' : LazyString(lambda:'API Documentation For Deep Learning'),
        'version' : LazyString(lambda:'1.0.0'),
        'description' : LazyString(lambda:'Dokumentasi API untuk Deep Learning'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers":[],
    "specs":[
        {
            "endpoint":'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path":"/flassger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


#inisiasi
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
sentiment = ["negative", "neutral", "positive"]

#fungsi cleansing text
def lowercase(text):
    return text.lower()

def hapus_karakter_ga_penting(text):
    text = re.sub('\n',' ',text) # Hapus enter
    text = re.sub('nya|deh|sih',' ',text) # Hapus stopwords tambahan
    text = re.sub('RT',' ',text) # Hapus RT
    text = re.sub('USER',' ',text) # Hapus USER
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ',text) # Hapus URL
    text = re.sub('  +', ' ', text) # Hapus extra spaces
    text = re.sub('[^a-zA-Z0-9]', ' ', text) #Hapus non huruf dan angka  
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text) #Hapus non huruf atau apostrophe
    text = ' '.join([w for w in text.split() if len(w)>1]) #Hapus huruf tunggal 
    return text
    
def hapus_nonhurufangka(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text) 
    return text

#mengambil file picke dan load model CNN
file = open("NeuralNetwork/x_pad_sequencesCNN.pickle",'rb')
feature_file_from_cnn = pickle.load(file)
file.close()

model_file_from_cnn = load_model("NeuralNetwork/modelCNN.h5")

#mengambil file picke dan load model LSTM
file = open("NeuralNetwork/x_pad_sequencesCNN.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model("NeuralNetwork/modelCNN.h5")

#CNN
@swag_from("yaml/cnn.yml", methods=['POST'])
@app.route('/cnn-teks', methods=['POST'])

def cnn():
    original_text = request.form.get('text')
    text = [hapus_nonhurufangka(hapus_karakter_ga_penting(lowercase(original_text)))]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using CNN",
        'data' : {
            'text': text,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("yaml/cnn_file.yml", methods=['POST'])
@app.route('/cnn-file', methods=['POST'])

def cnn_file():
    
    file = request.files['file']

    df = pd.read_csv(file, encoding='utf-8')

    kolom_text = df.iloc[:, 0]

    text_clean = []
    get_sentiment = []

    for text in kolom_text:
        text_clean_test = hapus_nonhurufangka(hapus_karakter_ga_penting(lowercase(text)))

        feature = tokenizer.texts_to_sequences(text_clean_test)
        feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

        prediction = model_file_from_cnn.predict(feature)
        get_sentiment_predict = sentiment[np.argmax(prediction[0])]

        text_clean.append(text_clean_test)
        get_sentiment.append(get_sentiment_predict)

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using CNN",
        'data' : {
            'text': text_clean,
            'sentiment' : get_sentiment,
        },
    }

    response_data = jsonify(json_response)
    return response_data

#LSTM
@swag_from("yaml/lstm.yml", methods=['POST'])
@app.route('/lstm-teks', methods=['POST'])

def lstm():
    original_text = request.form.get('text')
    text = [hapus_nonhurufangka(hapus_karakter_ga_penting(lowercase(original_text)))]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using CNN",
        'data' : {
            'text': text,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("yaml/lstm_file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])

def lstm_file():
    
    file = request.files['file']

    df = pd.read_csv(file, encoding='utf-8')

    kolom_text = df.iloc[:, 0]

    text_clean = []
    get_sentiment = []

    for text in kolom_text:
        text_clean_test = hapus_nonhurufangka(hapus_karakter_ga_penting(lowercase(text)))

        feature = tokenizer.texts_to_sequences(text_clean_test)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

        prediction = model_file_from_lstm.predict(feature)
        get_sentiment_predict = sentiment[np.argmax(prediction[0])]

        text_clean.append(text_clean_test)
        get_sentiment.append(get_sentiment_predict)

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using CNN",
        'data' : {
            'text': text_clean,
            'sentiment' : get_sentiment,
        },
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == "__main__":
    app.run(debug=True)

