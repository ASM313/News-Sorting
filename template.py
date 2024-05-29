import os
from pathlib import Path

# Complete Directory of project structure

project_name = "news_classification"

list_of_files = [
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transforamation.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/constants/database.py",
    f"{project_name}/constants/s3_bucket.py",
    f"{project_name}/constants/application.py",
    f"{project_name}/constants/env_variable.py",
    f"{project_name}/constants/training_pipeline/__init__.py",
    # f"{project_name}/constants/training_pipeline/",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/exception.py",
    f"{project_name}/logger.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/train_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/ml/__init__.py",    
    f"{project_name}/ml/model/__init__.py",
    f"{project_name}/ml/model/estimator.py",
    f"{project_name}/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/news_data.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/s3_syncer.py",
    f"notebooks/EDA.ipynb",
    f"notebooks/Model_build.ipynb",
    f"config/schema.yaml",
    f"templates/index.html",
    "app.py",
    "demo.py",
    "requirements.txt",
    "Dockerfile",
    "setup.py",
    ".dockerignore"
]




for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        # logging.info(f"Creating directory; {filedir} for the file: {filename}")
        print(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            # logging.info(f"Creating empty file: {filepath}")
            print(f"Creating empty file: {filepath}")

    else:
        # logging.info(f"{filename} is already exists")
        print((f"{filename} is already exists"))