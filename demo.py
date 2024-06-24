path = "artifact\06_02_2024_10_33_48\model_trainer\trained_model\model.pkl"
from news_classification.utils.main_utils import load_object

class ModelPrediction:


    def __init__(self,                    data_transformation_artifact:DataTransformationArtifact,model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
        except Exception as e:
            raise NewsException(e,sys)
    


    def initiate_model_prediction(self):
        try:
            transformed_object_file_path = self.data_transformation_artifact.transformed_object_file_path
            
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path

            # load preprocessor.pkl
            preprocessor = load_object(transformed_object_file_path)
            model = load_object(trained_model_file_path)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True