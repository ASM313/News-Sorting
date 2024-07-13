import os,sys
import dill
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from news_classification.constants.training_pipeline import TARGET_COLUMN
from news_classification.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from news_classification.entity.config_entity import DataTransformationConfig
from news_classification.exception import NewsException
from news_classification.logger import logging
# from news_classification.ml.model.estimator import TargetValueMapping
from news_classification.utils.main_utils import save_csv_data, save_object
from news_classification.constants.training_pipeline import SAVED_MODEL_DIR
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise NewsException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NewsException(e, sys)


    @classmethod
    def text_cleaning(self, words):

        try:
            
            # Let's apply stemming and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            return words 

        except Exception as e:
            raise NewsException(e, sys) from e

    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            label = LabelEncoder()
            tfidf = TfidfVectorizer(max_features=5000)
            
            # Label Encoding for target variables

            #training dataframe
            # input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            logging.info("Vectorizing train data")

            train_df['Text']=train_df['Text'].apply(self.text_cleaning) 
            train_df[TARGET_COLUMN] = label.fit_transform(train_df[TARGET_COLUMN])

            preprocessor_object = tfidf.fit(train_df['Text'])
            train_text_vectors = preprocessor_object.transform(train_df['Text']).toarray()
            dill.dump(label, open("label.pkl", "wb"))
            print("Shape of train data vectors: ",train_text_vectors.shape)
            
            #testing dataframe
            # input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            logging.info("Vectorizing test data")
            
            test_df['Text']=test_df['Text'].apply(self.text_cleaning)
            test_df[TARGET_COLUMN] = label.transform(test_df[TARGET_COLUMN])
            test_text_vectors = preprocessor_object.transform(test_df['Text']).toarray()

            print("Shape of test data vectors: ",test_text_vectors.shape)
            
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object)            
            dill.dump(preprocessor_object, open("preprocessor.pkl", "wb"))

            train_arr = train_df
            test_arr = test_df

        #     #save numpy array data
            transformed_data= pd.concat([train_arr, test_arr])
            save_csv_data(self.data_transformation_config.transformed_data_file_path,data=transformed_data)

         #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
            transformed_data_file_path=self.data_transformation_config.transformed_data_file_path,transformed_object_file_path=self.data_transformation_config.transformed_object_file_path)
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise NewsException(e, sys) from e