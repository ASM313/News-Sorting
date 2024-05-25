import sys
from typing import Optional

import numpy as np
import pandas as pd

from news_classification.configuration.mongo_db_connection import MongoDBClient
from news_classification.constants.database import DATABASE_NAME
from news_classification.exception import NewsException


class News_Data:
    """
    This class help to export entire MongoDB records as pandas dataframe
    """

    def __init__(self):
        """
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise NewsException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            # if database_name is None:
            #     collection = self.mongo_client.database[collection_name]
            # else:
            #     collection = self.mongo_client[database_name][collection_name]
            collection = self.mongo_client.database[collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            # Replace word 'na' with {Not a Number}
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise NewsException(e, sys)