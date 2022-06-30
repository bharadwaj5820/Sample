import numpy as np
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
class Best_Models:
    def __init__(self):
        pass
    def LinearRegression(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lin_regressor=LinearRegression()
        self.lin_regressor.fit(self.X_train,self.y_train)
        self.y_pred=self.lin_regressor.predict(self.X_test)
        self.accuracy_train_Lin=(r2_score(self.y_test,self.y_pred))*100
        self.accuracy_pred_Lin = (r2_score(self.y_train, self.lin_regressor.predict(self.X_train))) * 100
        return self.accuracy_train_Lin,self.accuracy_pred_Lin,self.lin_regressor
    def Svr(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.param_grid_SVR = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3,4],
            'gamma': [1e-4, 1e-3, 0.01, 0.1], 'C': [1, 10, 100]
        }
        self.grid=GridSearchCV(SVR(),self.param_grid_SVR,verbose=3,cv=5)
        self.grid.fit(self.X_train,self.y_train)
        self.kernel=self.grid.best_params_["kernel"]
        self.degree=self.grid.best_params_["degree"]
        self.gamma=self.grid.best_params_["gamma"]
        self.C=self.grid.best_params_["C"]
        self.SVR=SVR(kernel=self.kernel,degree=self.degree,gamma=self.gamma,C=self.C)
        self.y_pred=self.SVR.predict(self.X_test)
        self.accuracy_pred_SVR=r2_score(self.y_test,self.y_pred)*100
        self.accuracy_train_SVR=r2_score(self.y_train,self.SVR.predict(self.X_train))*100
        return self.accuracy_pred_SVR,self.accuracy_train_SVR,self.SVR
    def DecisionTree(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.param_grid_Decision={'Criterion':['gini', 'entropy'],'max_depth':np.linspace(1,32,32),
                                  'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True),
                                  'min_samples_leaf' : np.linspace(0.1, 0.5, 5, endpoint=True),
                                  'max_features': ['auto', 'log2']}

        self.Decision_tree_regressor=RandomizedSearchCV(estimator=DecisionTreeRegressor(), param_distributions=self.param_grid_Decision, n_iter=10, scoring='neg_mean_absolute_error', cv=3, verbose=2, random_state=42,refit=True)
        self.Decision_tree_regressor.fit(self.X_train,self.y_train)
        self.y_pred = self.Decision_tree_regressor.predict(self.X_test)
        self.accuracy_pred_DT = r2_score(self.y_test, self.y_pred) * 100
        self.accuracy_train_DT = r2_score(self.y_train, self.Decision_tree_regressor.predict(self.X_train)) * 100
        return self.accuracy_pred_DT, self.accuracy_train_DT,self.Decision_tree_regressor
    def Randomforest(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.param_grid_rf = {"n_estimators": [10, 50, 100, 130],
                           "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
        self.Randomforest_regressor=RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=self.param_grid_rf, n_iter = 10, scoring='neg_mean_absolute_error', cv = 3, verbose=2, random_state=42,return_train_score=True)
        self.Randomforest_regressor.fit(self.X_train,self.y_train)
        self.y_pred = self.Randomforest_regressor.predict(self.X_test)
        self.accuracy_pred_RF = r2_score(self.y_test, self.y_pred) * 100
        self.accuracy_train_RF = r2_score(self.y_train, self.Randomforest_regressor.predict(self.X_train)) * 100
        return self.accuracy_pred_RF, self.accuracy_train_RF,self.Randomforest_regressor
    def XGboost(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.param_grid_xgboost = {
            'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        }
        self.XGboost_regressor=RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=self.param_grid_xgboost,n_iter=10,scoring='neg_mean_absolute_error',cv=3,verbose=2, random_state=42,return_train_score=True)
        self.XGboost_regressor.fit(self.X_train,self.y_train)
        self.y_pred = self.XGboost_regressor.predict(self.X_test)
        self.accuracy_pred_XG = r2_score(self.y_test, self.y_pred) * 100
        self.accuracy_train_XG = r2_score(self.y_train, self.XGboost_regressor.predict(self.X_train)) * 100
        return self.accuracy_pred_XG, self.accuracy_train_XG,self.XGboost_regressor
    def bestmodel(self,X_train,y_train,X_test,y_test):
        self.accuracy_pred_XG, self.accuracy_train_XG,self.XGboost_regressor=self.XGboost(X_train,y_train,X_test,y_test)
        self.accuracy_pred_RF, self.accuracy_train_RF, self.Randomforest_regressor=self.Randomforest(X_train,y_train,X_test,y_test)
        if self.accuracy_pred_XG<self.accuracy_pred_RF:
            return "Random Forest",self.Randomforest_regressor
        else:
            return "XG Boost",self.XGboost_regressor

