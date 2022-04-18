#   -*- coding:utf-8 -*-
#   The main.py.py in my_hands_ml
#   created by Jiang Feng(silencejiang@zju.edu.cn)
#   created at 17:13 on 2022/4/15
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ch2.data import Data
from utils.logger import Logger

class Ch2:
    def __init__(self):
        self.log = Logger("mh").get_log()
        d =Data()
        # d.fetch_housing_data()
        self.housing = d.load_housing_data()
        self.data = None
        self.IMAGES_PATH = None
        self.train_set = None
        self.test_set = None
        self.housing_labels =None
        self.housing_num = None
        self.housing_prepared = None
        self.full_pipeline = None
        self.final_model = None

    def init_pic_dir(self):
        PROJECT_ROOT_DIR = ".."
        CHAPTER_ID = "end_to_end_project"
        IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
        os.makedirs(IMAGES_PATH, exist_ok=True)
        self.IMAGES_PATH = IMAGES_PATH

    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(self.IMAGES_PATH, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)

    def describe(self):
        self.log.info(self.housing.head())
        self.log.info(self.housing.info())
        self.log.info(self.housing["ocean_proximity"].value_counts())
        self.log.info(self.housing.describe())
        self.housing.hist(bins=50, figsize=(20, 15))
        self.save_fig("attribute_histogram_plots")

    def create_test_set(self, stratified=True):
        if not stratified:
            from sklearn.model_selection import train_test_split
            self.train_set ,self.test_set= train_test_split(self.housing ,test_size= 0.2 ,random_state= 42)
        else:
            self.housing["income_cat"] = pd.cut(self.housing["median_income"],
                                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                           labels=[1, 2, 3, 4, 5])
            # self.log.info(self.housing["income_cat"])
            from sklearn.model_selection import StratifiedShuffleSplit
            split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2, random_state= 42)
            for train_index , test_index in split.split(self.housing, self.housing["income_cat"]):
                self.train_set = self.housing.loc[train_index]
                self.test_set = self.housing.loc[test_index]

            self.log.info(self.test_set["income_cat"].value_counts())
            for set_ in (self.train_set, self.test_set):
                set_.drop("income_cat", axis=1, inplace=True)

    def show_data(self):
        self.housing = self.train_set.copy()

        self.housing.plot(kind="scatter", x="longitude", y="latitude")
        self.save_fig("bad_visualization_plot")

        self.housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
        self.save_fig("better_visualization_plot")

        self.housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                     s=self.housing["population"] / 100, label="population", figsize=(10, 7),
                     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                     sharex=False)
        plt.legend()
        self.save_fig("housing_prices_scatterplot")

    def show_corr(self):
        corr_matrix = self.housing.corr()
        self.log.info(corr_matrix["median_house_value"].sort_values(ascending=False))

        from pandas.plotting import scatter_matrix

        attributes = ["median_house_value", "median_income", "total_rooms",
                      "housing_median_age"]
        scatter_matrix(self.housing[attributes], figsize=(12, 8))
        self.save_fig("scatter_matrix_plot")

        self.housing.plot(kind="scatter", x="median_income", y="median_house_value",
                     alpha=0.1)
        plt.axis([0, 16, 0, 550000])
        self.save_fig("income_vs_house_value_scatterplot")

    def add_new_features(self):
        self.housing["rooms_per_household"] = self.housing["total_rooms"] / self.housing["households"]
        self.housing["bedrooms_per_room"] = self.housing["total_bedrooms"] / self.housing["total_rooms"]
        self.housing["population_per_household"] = self.housing["population"] / self.housing["households"]


    def reset(self):
        self.housing = self.train_set.drop("median_house_value", axis=1)  # drop labels for training set
        self.housing_labels = self.train_set["median_house_value"].copy()

    def data_cleaning(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy= "median")
        housing_num = self.housing.drop('ocean_proximity', axis=1)
        self.housing_num = housing_num
        imputer.fit(housing_num)
        self.log.info(imputer.statistics_)
        self.log.info(housing_num.median().values)

        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                                  index=self.housing.index)
        sample_incomplete_rows = self.housing[self.housing.isnull().any(axis=1)].head()
        self.log.info(housing_tr.loc[sample_incomplete_rows.index.values])

    def text_process(self):
        housing_cat = self.housing[['ocean_proximity']]
        self.log.info(housing_cat.head(10))

        from sklearn.preprocessing import OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
        self.log.info(housing_cat_encoded[:10])
        self.log.info(ordinal_encoder.categories_)

        from sklearn.preprocessing import OneHotEncoder
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        self.log.info(type(housing_cat_1hot))
        self.log.info(housing_cat_1hot)
        self.log.info(housing_cat_1hot.toarray())

        cat_encoder = OneHotEncoder(sparse=False)
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        self.log.info(type(housing_cat_1hot))
        self.log.info(housing_cat_1hot)

    def add_extra_features(self, X, add_bedrooms_per_room=True):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def my_transform(self):
        from ch2.transformer import CombinedAttributesAdder

        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(self.housing.values)
        self.log.info(housing_extra_attribs)

        housing_extra_attribs = pd.DataFrame(
            housing_extra_attribs,
            columns=list(self.housing.columns) + ["rooms_per_household", "population_per_household"],
            index=self.housing.index)
        self.log.info(housing_extra_attribs.head())

        from sklearn.preprocessing import FunctionTransformer
        attr_adder = FunctionTransformer(self.add_extra_features, validate=False,
                                         kw_args={"add_bedrooms_per_room": False})
        housing_extra_attribs = attr_adder.fit_transform(self.housing.values)
        self.log.info(housing_extra_attribs)

    def my_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.preprocessing import StandardScaler

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', FunctionTransformer(self.add_extra_features, validate=False)),
            ('std_scaler', StandardScaler()),
        ])

        housing_num_tr = num_pipeline.fit_transform(self.housing_num)
        self.log.info(housing_num_tr)

        from ch2.transformer import CombinedAttributesAdder

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        housing_num_tr = num_pipeline.fit_transform(self.housing_num)
        self.log.info(housing_num_tr)

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        num_attribs = list(self.housing_num)
        cat_attribs = ["ocean_proximity"]

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        housing_prepared = full_pipeline.fit_transform(self.housing)

        self.log.info(housing_prepared)
        self.log.info(housing_prepared.shape)
        self.housing_prepared = housing_prepared
        self.full_pipeline = full_pipeline

    def train_models(self):
        from sklearn.linear_model import LinearRegression

        lin_reg = LinearRegression()
        lin_reg.fit(self.housing_prepared, self.housing_labels)

        some_data = self.housing.iloc[:5]
        some_labels = self.housing_labels.iloc[:5]
        some_data_prepared = self.full_pipeline.transform(some_data)

        self.log.info("Predictions:" +str(lin_reg.predict(some_data_prepared)))
        self.log.info("Labels:" +str(list(some_labels)))
        self.log.info("some_data_prepared:" +str(some_data_prepared))

        from sklearn.metrics import mean_squared_error

        housing_predictions = lin_reg.predict(self.housing_prepared)
        lin_mse = mean_squared_error(self.housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)
        self.log.info(lin_rmse)

        from sklearn.metrics import mean_absolute_error

        lin_mae = mean_absolute_error(self.housing_labels, housing_predictions)
        self.log.info(lin_mae)

        from sklearn.tree import DecisionTreeRegressor

        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(self.housing_prepared, self.housing_labels)

        housing_predictions = tree_reg.predict(self.housing_prepared)
        tree_mse = mean_squared_error(self.housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)
        self.log.info(tree_rmse)

        # Fine-tune your model
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(tree_reg, self.housing_prepared, self.housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        self.display_scores(tree_rmse_scores)

        lin_scores = cross_val_score(lin_reg, self.housing_prepared, self.housing_labels,
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        self.display_scores(lin_rmse_scores)

        from sklearn.ensemble import RandomForestRegressor

        forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
        forest_reg.fit(self.housing_prepared, self.housing_labels)

        housing_predictions = forest_reg.predict(self.housing_prepared)
        forest_mse = mean_squared_error(self.housing_labels, housing_predictions)
        forest_rmse = np.sqrt(forest_mse)
        self.log.info(forest_rmse)

        forest_scores = cross_val_score(forest_reg, self.housing_prepared, self.housing_labels,
                                        scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        self.display_scores(forest_rmse_scores)


    def display_scores(self, scores):
        self.log.info("Scores:" +str(scores))
        self.log.info("Mean:" + str(scores.mean()))
        self.log.info("Standard deviation:" + str(scores.std()))


    def fine_tune(self):
        from sklearn.model_selection import  GridSearchCV
        from sklearn.ensemble import RandomForestRegressor

        para_grid =[
            {"n_estimators":[3,10,30], "max_features":[2,4,6,8]},
            {"bootstrap":[False],"n_estimators":[3,10],"max_features":[2,3,4]},
        ]

        forest_reg = RandomForestRegressor(random_state= 42)
        grid_search = GridSearchCV(forest_reg, para_grid,cv=5,scoring="neg_mean_squared_error",return_train_score=True)
        grid_search.fit(self.housing_prepared, self.housing_labels)

        self.log.info(grid_search.best_estimator_)
        feature_importances = grid_search.best_estimator_.feature_importances_
        self.log.info(feature_importances)

        self.final_model = grid_search.best_estimator_

        # extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
        # # cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
        # cat_encoder = self.full_pipeline.named_transformers_["cat"]
        # cat_one_hot_attribs = list(cat_encoder.categories_[0])
        # attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        # sorted(zip(feature_importances, attributes), reverse=True)

        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

        self.log.info(pd.DataFrame(grid_search.cv_results_))

        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint

        param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }

        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(self.housing_prepared, self.housing_labels)
        self.log.info(rnd_search.best_estimator_)

    def test(self):
        X_test = self.test_set.drop("median_house_value", axis=1)
        y_test = self.test_set["median_house_value"].copy()

        X_test_prepared = self.full_pipeline.transform(X_test)
        final_predictions = self.final_model.predict(X_test_prepared)

        from sklearn.metrics import mean_squared_error
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        self.log.info(final_rmse)

    def save(self):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        full_pipeline_with_predictor = Pipeline([
            ("preparation", self.full_pipeline),
            ("linear", LinearRegression())
        ])
        my_model = full_pipeline_with_predictor

        import joblib
        joblib.dump(my_model, "my_model.pkl")
        my_model_loaded = joblib.load("my_model.pkl")

        my_model_loaded.fit(self.housing, self.housing_labels)
        some_data = self.housing.iloc[:5]
        self.log.info(my_model_loaded.predict(some_data))

if __name__ == "__main__":
    c = Ch2()
    c.init_pic_dir()
    # c.describe()
    c.create_test_set()
    # c.show_data()
    # c.show_corr()
    c.add_new_features()
    c.reset()
    c.data_cleaning()
    c.text_process()
    c.my_transform()
    c.my_pipeline()
    c.train_models()
    c.fine_tune()
    c.test()
    c.save()