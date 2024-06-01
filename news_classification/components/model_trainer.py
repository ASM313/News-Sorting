from news_classification.utils.main_utils import load_numpy_array_data
from news_classification.exception import NewsException
from news_classification.logger import logging
from news_classification.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from news_classification.entity.config_entity import ModelTrainerConfig
import os,sys
from sklearn.feature_extraction.text import TfidfVectorizer 
from news_classification.ml.model.estimator import NewsModel
from news_classification.utils.main_utils import save_object,load_object
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
from news_classification.constants.training_pipeline import TARGET_COLUMN
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NewsException(e,sys)

    def perform_hyper_paramter_tunig(self):...
    
    def train_model(self,x_train,y_train):
        try:
            logging.info("Model object created")
            mnb_clf = MultinomialNB()

            logging.info("Fitting in model")
            mnb_clf.fit(x_train,y_train)
            return mnb_clf
        except Exception as e:
            raise e
    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            transformed_data_file_path = self.data_transformation_artifact.transformed_data_file_path
            transformed_object_file_path = self.data_transformation_artifact.transformed_object_file_path
            
            # Load data into dataframe
            df = pd.read_csv(transformed_data_file_path)
            
            # define features and variables
            features = df.drop(columns=[TARGET_COLUMN], axis = 1)
            target = df[[TARGET_COLUMN]]

            # load preprocessor.pkl

            preprocessor = load_object(transformed_object_file_path)
            preprocessor.fit(features['Text'])

            X = preprocessor.transform(features['Text']).toarray()

            x_train, x_test, y_train, y_test = train_test_split(X, target, test_size = 0.2, random_state=42)

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)     
            
            y_test_pred = model.predict(x_test)

            print("Accuracy: ",accuracy_score(y_test_pred, y_test))
            print("Precision: ",precision_score(y_test, y_test_pred, average='weighted'))
            # print("Recall: ",recall_score(y_test_pred, y_test))
            # print("F1-score: ",f1_score(y_test_pred, y_test))
     
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            news_model = NewsModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=news_model)

            # Model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
        except Exception as e:
            raise NewsException(e,sys)