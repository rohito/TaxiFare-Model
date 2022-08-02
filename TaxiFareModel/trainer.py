from sklearn.pipeline import Pipeline
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

class Trainer():

    def __init__(self):
        pass

    def holdout(self):
        df = get_data()
        df = clean_data(df)
        y = df["fare_amount"]
        X = df.drop("fare_amount",axis=1)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.15)

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
        self.pipe = Pipeline([
            ('preproc',preproc_pipe),
            ('linear_model',LinearRegression())
        ])
        return self.pipe

    def run(self):
        self.holdout()
        self.set_pipeline()
        self.pipe.fit(self.X_train,self.y_train)


    def evaluate(self):
        y_pred = self.pipe.predict(self.X_val)
        rmse = compute_rmse(y_pred,self.y_val)
        print(rmse)
        return rmse

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
    rmse = trainer.evaluate()
    print(rmse)
