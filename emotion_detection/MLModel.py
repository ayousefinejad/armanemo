from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from preprocess import tfidf_vectorizer
from configuration import BaseConfig
from models import ML_Models


if __name__ == '__main__':
    config = BaseConfig().get_config()

    DATA = pd.read_csv(config.processed_data_dir)
    DATA.dropna(axis=0, inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(DATA['text'], DATA['label'], test_size=0.2, random_state=42)
    X_train_tfidf, X_val_tfidf = tfidf_vectorizer(X_train, X_val)

    model = ML_Models(X_train_tfidf, X_val_tfidf, y_train, y_val, classes=np.unique(y_train))
    model.train_models(cv=True)

    model.plot_cv_results()

    random_forest_model = model.models['Random Forest']
    svm_model = model.models['SVM']
    catboost_model = model.models['CatBoost']

    model.plot_confusion_matrix(random_forest_model)
    model.plot_confusion_matrix(svm_model)
    model.plot_confusion_matrix(catboost_model)


