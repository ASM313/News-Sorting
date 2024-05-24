from news_classification.configuration.mongo_db_connection import MongoDBClient
from news_classification.exception import NewsException
import os,sys
from news_classification.logger import logging

# def test():
#     try:
#         logging.info("Testing")
#         a=1/0
    
#     except Exception as e:
#         raise NewsException(e,sys)


if __name__ == '__main__':
    mongodb_client = MongoDBClient()
    print("Database :", mongodb_client.database_name)
    # test()