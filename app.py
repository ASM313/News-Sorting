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
from demo import ModelPrediction
from news_classification.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

obj = ModelPrediction(DataTransformationArtifact(), ModelTrainerArtifact())


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

    
@app.route("/predict", methods=['GET', 'POST'])
def predict_route():
    pass

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 