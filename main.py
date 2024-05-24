from news_classification.configuration.mongo_db_connection import MongoDBClient

if __name__ == '__main__':
    mongodb_client = MongoDBClient()
    # print("Collection :", mongodb_client.database)
    print("Collection :", mongodb_client.database_name)