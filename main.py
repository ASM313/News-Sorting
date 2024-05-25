from news_classification.configuration.mongo_db_connection import MongoDBClient
from news_classification.exception import NewsException
import os,sys
from news_classification.logger import logging
from news_classification.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from news_classification.pipeline.train_pipeline import TrainPipeline

# def test():
#     try:
#         logging.info("Testing")
#         a=1/0
    
#     except Exception as e:
#         raise NewsException(e,sys)


if __name__ == '__main__':


    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
    # training_pipeline_config = TrainingPipelineConfig()
    # data_ingestion_config = DataIngestionConfig(
    #                         training_pipeline_config=training_pipeline_config
    #                         )
    
    # mongodb_client = MongoDBClient()
    # print("Database :", mongodb_client.database_name)
    # test()