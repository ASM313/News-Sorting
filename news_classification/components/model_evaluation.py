from news_classification.exception import NewsException
from news_classification.logger import logging
from news_classification.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from news_classification.entity.config_entity import ModelEvaluationConfig
import os,sys
from news_classification.ml.metric.classification_metric import get_classification_score
from news_classification.ml.model.estimator import NewsModel
from news_classification.utils.main_utils import save_object,load_object,write_yaml_file
from news_classification.ml.model.estimator import ModelResolver
from news_classification.constants.training_pipeline import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
import pandas  as  pd
from sklearn.model_selection import train_test_split

class ModelEvaluation:


    def __init__(self,model_eval_config:ModelEvaluationConfig,
                    data_transformation_artifact:DataTransformationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            self.model_eval_config=model_eval_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise NewsException(e,sys)
    


    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
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


            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True


            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)
            
            y_trained_pred = train_model.predict(x_test)
            y_latest_pred  =latest_model.predict(x_test)

            trained_metric = get_classification_score(y_test, y_trained_pred)
            latest_metric = get_classification_score(y_test, y_latest_pred)

            improved_accuracy = trained_metric.f1_score-latest_metric.f1_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                #0.02 < 0.03
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)

            model_eval_report = model_evaluation_artifact.__dict__

            #save the report
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise NewsException(e,sys)

    
    