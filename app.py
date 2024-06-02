from news_classification.configuration.mongo_db_connection import MongoDBClient
from news_classification.exception import NewsException
import os,sys
from news_classification.logger import logging
from news_classification.pipeline import training_pipeline
from news_classification.pipeline.training_pipeline import TrainPipeline
import os
from news_classification.utils.main_utils import read_yaml_file
from news_classification.constant.training_pipeline import SAVED_MODEL_DIR
from news_classification.constant.application import APP_HOST, APP_PORT
from news_classification.ml.model.estimator import ModelResolver,TargetValueMapping
from news_classification.utils.main_utils import load_object
from flask import Flask,request,render_template
import pandas as pd
import numpy as np


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

    
@app.route("/predict", methods=['GET', 'POST'])
def predict_route():
    try:
        logging.info("User request for Prediction..")    

        #get data from user csv file
        csv_file = request.files['csvFile']
        logging.info("User uploaded csv file.")

        if csv_file.filename == '':
            return 'No selected file'

        if csv_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
            csv_file.save(file_path)
            logging.info('File uploaded successfully')

        #convert csv file to dataframe
        user_input_csv_df = pd.read_csv(file_path)
        logging.info("CSV to Dataframe")
        print(user_input_csv_df)
        
        # Convert dataframe to numpy array
        user_input_csv_array = np.array(user_input_csv_df)
        logging.info("Dataframe to np array ")
        
        print("User input : ", user_input_csv_array)
        
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            msg = "Model is not available"
            return render_template("index.html", msg=msg)
        
        best_model_path = model_resolver.get_best_model_path()
        logging.info("Best model chosen..")

        model = load_object(file_path=best_model_path)
        logging.info("Model loaded...")

        y_pred = model.predict(user_input_csv_array)
        print("Predicted class is :", y_pred[0])
        logging.info("Predicted class showed:")
        
        logging.info("Appending predicted column with dataframe.")
        user_input_csv_df['predicted_class'] = y_pred
        user_input_csv_df['predicted_class'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        
        #decide how to return file to user.
        """
        Write logic here
        """

        if  y_pred[0]==1:
            result = "Positive"

        else:
            result = "Negative"    
        logging.info("Showing result to User at Front end")

        return render_template("index.html", result=result)
        
    except Exception as e:
        err_msg = f"Error Occurred! {e}"
        return render_template("error.html", msg=err_msg)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 