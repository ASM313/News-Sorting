from news_classification.constants.training_pipeline import SAVED_MODEL_DIR ,MODEL_FILE_NAME
import os
from sklearn.feature_extraction.text import TfidfVectorizer
class TargetValueMapping:
    def __init__(self):
        # self.neg: int = 0
        # self.pos: int = 1
        self.sport         : int = 0 
        self.business      : int = 1
        self.politics      : int = 2
        self.entertainment : int = 3
        self.tech          : int = 4

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

tfidf = TfidfVectorizer()

# Write a code to train model and check the accuracy.


class NewsModel:

    def __init__(self,model):
        try:
            self.model = model
        except Exception as e:
            raise e

    def text_cleaning(words):
    
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

    def predict(self, x):
        try:
            # x_cleaned = self.text_cleaning(x)
            # x_transformed = `
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e
    

class ModelResolver:

    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e

    def get_best_model_path(self,)->str:
        try:
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e

    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise e