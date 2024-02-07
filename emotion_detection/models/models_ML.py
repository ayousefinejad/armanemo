import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, make_scorer, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from catboost import CatBoostClassifier

logging.basicConfig(filename=f'/Users/arshiayousefi/Desktop/ML/armanemo/assets/logs/ML_Models.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ML_Models:
    def __init__(self, X_train, X_test, y_train, y_test, classes):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classes = classes  # Add classes attribute

        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),  # Probability=True might be needed for some metrics
            'CatBoost': CatBoostClassifier(random_seed=42, verbose=False),  # You can adjust parameters as needed
        }

        self.cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        self.scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='macro', zero_division=0),
            # Adjust for multiclass and handle undefined precision
            'recall': make_scorer(recall_score, average='macro', zero_division=0),  # Handle undefined recall
            'f1': make_scorer(f1_score, average='macro', zero_division=0)
        }

        self.cv_results = {model_name: None for model_name in self.models}

    def _grid_search(self, model, param_grid):
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv,
            refit='accuracy',  # Specify the metric for refitting
        )
        grid_search.fit(self.X_train, self.y_train)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _cross_validation(self, model, param_grid, model_name):
        best_model = self._grid_search(model, param_grid)
        results = cross_validate(best_model, self.X_train, self.y_train, cv=self.cv, scoring=self.scoring)

        logging.info(f'Cross-validation results for {model_name}')
        for metric in self.scoring:
            logging.info(f'Mean {metric.capitalize()}: {np.mean(results[f"test_{metric}"])}')

        return best_model, results

    def plot_confusion_matrix(self, model):
        y_pred = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred, labels=self.classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
        disp.plot()
        plt.title(f'Confusion Matrix for {model.__class__.__name__}')
        plt.savefig(f'/Users/arshiayousefi/Desktop/ML/armanemo/reports/confusion_matrix_{model.__class__.__name__}.png')
        # plt.show()

    def _evaluation(self, model):
        y_pred = model.predict(self.X_test)
        logging.info(f'accuracy {model.__class__.__name__}: {accuracy_score(self.y_test, y_pred)}')

    def train_models(self, cv=False):
        param_grids = {
            'Random Forest': {
                'n_estimators': [10, 30, 100],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5]},
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1]},
            'CatBoost': {
                'iterations': [100, 300, 500],
                'learning_rate': [0.01, 0.1, 0.3],
                'depth': [4, 6, 10]
            }
        }
        

        for model_name, model in self.models.items():
            logging.info(f'\n {model_name} model ')
            if cv == True:
                self.models[model_name], self.cv_results[model_name] = self._cross_validation(model,
                                                                                              param_grids[model_name],
                                                                                              model_name)
            else:
                self.models[model_name] = model.fit(self.X_train, self.y_train)
                self._evaluation(model)

    def plot_cv_results(self, metric='accuracy'):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y=f'test_{metric}', data=self._get_cv_results_df())
        plt.title(f'Cross-Validation {metric.capitalize()} Scores for Each Model')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.savefig('/Users/arshiayousefi/Desktop/ML/armanemo/reports/ML_Models_cv_results.png')
        plt.show()

    def _get_cv_results_df(self):
        dfs = []
        for model_name, cv_result in self.cv_results.items():
            df = pd.DataFrame(cv_result)
            df['Model'] = model_name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
