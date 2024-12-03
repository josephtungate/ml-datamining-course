import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV

#To make it easier to consistantly load the csvs across all tasks.
def load_gwp(path):
    return (np.genfromtxt(path, dtype=str, delimiter=',', skip_header=1, encoding="utf-8"),  
            np.genfromtxt(path, dtype=str, delimiter=',', max_rows=1, encoding="utf-8").astype(str))

def load_star(path):
    return (np.genfromtxt(path, dtype=str, delimiter=',', skip_header=1, encoding="utf-8"),
           np.genfromtxt(path, dtype=str, delimiter=',', max_rows=1, encoding="utf-8").astype(str))


#Maps date given as MM/DD/YYYY to ISO format.
def US_to_ISO_date(date):
    components = date.split('/')
    return components[2] + '-' + components[0] + '-' + components[1]

#Splits ISO date strings as arrays of years, months, and days.
def split_dates(datetimes, iso_converter=None, missing_value='', filling_value='nan'):
    datetimes = datetimes.copy()
    #Convert non-missing values to ISO format dates if converter is given.
    if(iso_converter != None):
        datetimes = np.array(list(map(
            lambda x: x if x == missing_value else iso_converter(x),
            datetimes)))
        
    split_dates = np.array(list(map(
        lambda x: [filling_value, filling_value, filling_value] if x == missing_value
                  else x.split('-'),
            datetimes)))    

    return np.array(split_dates)

def abs_r_regression(X, y):
    return np.abs(r_regression(X, y))
    

class GWPDataset:
    _raw_data = None
    _raw_feature_names = None
    _processed_data = None
    _processed_feature_names = None

    def __init__(self, path):
        self._raw_data, self._raw_feature_names = load_gwp(path)

    def _gwp_encoder(self, dataset, missing_value='', filling_value='nan'):
        #Impute blank strings (i.e. missing values) to NaN values for casting.
        dataset = SimpleImputer(missing_values=missing_value, strategy='constant', fill_value=filling_value).fit_transform(dataset.astype('object'))
        dates = split_dates(dataset[:, 0], iso_converter=US_to_ISO_date, missing_value=filling_value, filling_value=filling_value).astype('float64')
        ordinal_features = OrdinalEncoder().fit_transform(dataset[:, 1:4]).astype('float64') #Quarter, Department, and Day. 
        real_features = dataset[:, 4:].astype('float64')

        return np.hstack((dates, ordinal_features, real_features))

    def _process_data(self):
        data = self._gwp_encoder(self._raw_data)
        data = KNNImputer(missing_values=np.nan, n_neighbors=5).fit_transform(data)
        gwp_selector = SelectPercentile(score_func=abs_r_regression, percentile=50)
        data = np.hstack((gwp_selector.fit_transform(data[:, 1:-1], data[:, -1]), data[:, -1:]))
        self._processed_feature_names = gwp_selector.get_feature_names_out(input_features=np.concatenate(
            (np.array(["month", "day(date)"]), self._raw_feature_names[1:-1])))
        data = np.hstack((MinMaxScaler(feature_range=(0, 1)).fit_transform(data[:, :-1]),
                          data[:, -1:]))
        return data

    def raw_data(self):
        return self._raw_data

    def raw_feature_names(self):
        return self._raw_feature_names

    def processed_data(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data

    def processed_feature_names(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_feature_names
        
    def processed_X(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data[:, :-1]
        
    def processed_Y(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data[:, -1]

class StarDataset:
    _raw_data = None
    _raw_feature_names = None
    _processed_data = None
    _processed_feature_names = None

    def __init__(self, path):
        self._raw_data, self._raw_feature_names = load_star(path)

    def _star_encoder(self, dataset, missing_value='', filling_value='nan'):
        #Impute blank strings (i.e. missing values) to NaN values for casting.
        dataset = SimpleImputer(missing_values=missing_value, strategy='constant', fill_value=filling_value).fit_transform(dataset.astype('object'))
        class_feature = OrdinalEncoder().fit_transform(dataset[:, -1:]).astype('float64')
        real_features = dataset[:, 0:-1].astype('float64')
    
        return np.hstack((real_features, class_feature))
    

    def _process_data(self):
        data = self._star_encoder(self._raw_data)
        data = KNNImputer(missing_values=np.nan, n_neighbors=5).fit_transform(data)
        data = np.hstack((MinMaxScaler(feature_range=(0, 1)).fit_transform(data[:, :-1]), data[:, -1:]))
        star_selector = SelectPercentile(score_func=chi2, percentile=50)
        data = np.hstack((star_selector.fit_transform(data[:, 0:-1], data[:, -1:]), data[:, -1:]))
        self._processed_feature_names = star_selector.get_feature_names_out(input_features=self._raw_feature_names[0:-1])
        return data

    def raw_data(self):
        return self._raw_data

    def raw_feature_names(self):
        return self._raw_feature_names

    def processed_data(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data

    def processed_feature_names(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_feature_names
        
    def processed_X(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data[:, :-1]
        
    def processed_Y(self):
        if self._processed_data is None:
            self._processed_data = self._process_data()
        return self._processed_data[:, -1]
    
#model - estimator to optimise.
#dataset - object like GWPDataset or StarDataset
#param_grid - Passed to GridSearchCV's param_grid parameter. Controls parameter space.
#scoring - A dict mapping scorer names to scorer functions. Either functions ending with _score or those returned by make_scorer.
#cv - Passed to GridSearchCV's cv parameter. Controls splitting strategy.
#refit - If scoring is not None, then specifies the name in 'scoring' which denotes the metric by which to fit the final estimator.
#train_size - Size of the training set.
#stratify - Whether to stratify the dataset when splitting.
#shuffle - Whether to shuffle the dataset before splitting.
#random_state - Seed by which to shuffle. Defaults to 1 to allow same shuffling
#when evaluating different models on same dataset.
def train(model, dataset, param_grid, scoring=None, cv=None, refit=True,
          train_size=0.8, stratify=None, shuffle=True, random_state=1):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.processed_X(), dataset.processed_Y(),
        train_size=train_size, stratify=stratify, shuffle=shuffle,
        random_state=random_state)
    
    gridsearch = GridSearchCV(model, param_grid,
                              scoring=scoring, refit=refit, cv=cv)
    gridsearch.fit(x_train, y_train)
    
    best_model=gridsearch.best_estimator_
    
    #Evaluate giving scoring metrics on test data.
    test_scores = {}
    if(not(scoring is None)):
        for name, scorer in scoring.items():
            test_scores.update({name: scorer(best_model, x_test, y_test)})
    else:
        test_scores = {'score' : model.score(x_test, y_test)}
    return {"best_model": best_model,
            "grid_search": gridsearch,
            "test_scores": test_scores,
            "dataset": (x_train, x_test, y_train, y_test)}
