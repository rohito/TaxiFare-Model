from sklearn.pipeline import Pipeline
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[GB] [London] [Rohito] TaxiFareModel + 1.0"

class Trainer():


    def __init__(self,X,y):
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    # def data():
    #     df = get_data()
    #     df = clean_data(df)
    #     y = df["fare_amount"]
    #     X = df.drop("fare_amount",axis=1)
    #     return X,y

    # def holdout(self):
    #     self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.15)

    def set_pipeline(self):
        #Distance Pipeline
        dist_pipe = Pipeline([
            ('dist_trans',DistanceTransformer()),
            ('stdscaler',StandardScaler())
        ])
        #Time features Pipeline
        time_pipe = Pipeline([
            ('time_enc',TimeFeaturesEncoder('pickup_datetime')),
            ('ohe',OneHotEncoder(handle_unknown='ignore'))
        ])
        #Preprocessing Pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time',time_pipe,['pickup_datetime'])
        ],remainder='drop')
        #Model Pipeline
        self.pipeline = Pipeline([
            ('preproc',preproc_pipe),
            ('linear_model',LinearRegression())
        ])
        return self.pipeline

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        self.mlflow_log_param('model','Linear')


    def evaluate(self, X_test, y_test):

        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('rmse',rmse)
        self.mlflow_log_param( "student_name", "Rohit")
        print(rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        pass

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == '__main__':
    df = get_data()
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount",axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train,y_train)
    trainer.run()
    rmse = trainer.evaluate(X_val,y_val)
    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
    # client = trainer.mlflow_client
    # run = client.create_run(experiment_id)
    # client.log_metric(run.info.run_id, "rmse", rmse)
    # client.log_param(run.info.run_id, "model", "linear model")
