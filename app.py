from news_classification.configuration.mongo_db_connection import MongoDBClient
from news_classification.exception import NewsException
from news_classification.logger import logging
from news_classification.pipeline import train_pipeline
from news_classification.pipeline.train_pipeline import TrainPipeline
from news_classification.utils.main_utils import read_yaml_file
from news_classification.constants.training_pipeline import SAVED_MODEL_DIR
from news_classification.constants.application import APP_HOST, APP_PORT
from news_classification.ml.model.estimator import ModelResolver
from news_classification.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from news_classification.utils.main_utils import load_object
from news_classification.utils.main_utils import text_cleaning
import os,sys
import dill
from flask import Flask,request,render_template
import pandas as pd
import numpy as np

with open("preprocessor.pkl", "rb") as file:
    preprocessor = dill.load(file)
    
with open("model.pkl", "rb") as file:
    model = dill.load(file)
    
with open("label.pkl", "rb") as file:
    label = dill.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict_route():

    try:
        logging.info("User request for Prediction..")    

        #get data from user 
        text = request.form.get('user_msg')
        logging.info("User uploaded message")
        
        # Make a Dataframe
        data = {'Text': [text]}
        df = pd.DataFrame(data)
        print(df)
        
        # preprocess the user_msg 
        cleaned_text=text_cleaning(text)
        print(cleaned_text)
        
        # Create vectors
        preprocessed_text=preprocessor.transform([cleaned_text]).toarray()  
        print(preprocessed_text)     
        
        # Make Predictions
        prediction = model.predictions(np.array(preprocessed_text))
        prediction=label.inverse_transform(prediction)
        print("Predicted class is :", prediction)
        logging.info("Predicted class showed: ", prediction)
        
        return render_template("index.html", prediction=prediction)
        
    except Exception as e:
        err_msg = f"Error Occurred! {e}"
        return render_template("error.html", msg=err_msg)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 