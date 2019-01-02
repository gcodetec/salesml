import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

print(tf.__version__)

CSV_COLUMNS = ['quantidadetotal','produto','ano','mes']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
FEATURES.append('crescimento')
LABEL = CSV_COLUMNS[0]

df_main = pd.read_csv('./datasets/sales_main.csv', header = 0, names = CSV_COLUMNS)

df_main['crescimento'] = np.nan

for index, df in df_main.iterrows():
  produto = df['produto']
  mes = df['mes']
  ano = df['ano']

  if mes == 1 or mes == 1.0:
    mes = 13.0

  df_qtd_anterior = df_main.loc[(df_main.produto == produto) & (df_main.ano <= ano) & (df_main.mes < mes) , 'quantidadetotal']
  crescimento = 0.0

  if not df_qtd_anterior.empty:
    qtd_anterior = df_qtd_anterior.values[-1]
    crescimento = ((df['quantidadetotal'] - qtd_anterior) / qtd_anterior) * 100

  df_main.ix[index, ['crescimento']] = crescimento

print(df_main.head())

# df_main.drop('nome')
# df_train = pd.read_csv('./sales-train.csv', header = None, names = CSV_COLUMNS)
# df_valid = pd.read_csv('./sales-valid.csv', header = None, names = CSV_COLUMNS)
# df_test = pd.read_csv('./sales-test.csv', header = None, names = CSV_COLUMNS)

msk = np.random.rand(len(df_main)) < 0.8

df_train = df_main[msk]
test = df_main[~msk]
msk2 = np.random.rand(len(test)) < 0.5
df_test = test[msk2]
df_valid = test[~msk2]

# print( df_valid)
def make_train_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )

def make_eval_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df[LABEL],
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )

def make_prediction_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )

def make_feature_cols():
    input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    return input_columns

def predict(model, df):
    predictions = model.predict(input_fn = make_prediction_input_fn(df))

    print(df)

    for items in predictions:
        print(items)

def train_model(model):
    pass

def print_rmse(model, df):
    metrics = model.evaluate(input_fn = make_eval_input_fn(df))
    print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

def run_test_predict(model):
    print("============test predict============")
    predict(model, df_test)

def train_and_evaluate():
    pass

OUTDIR = 'sale_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.LinearRegressor(
      feature_columns = make_feature_cols(), model_dir = OUTDIR)

# model = tf.estimator.DNNRegressor(hidden_units = [32, 8, 2],
#     feature_columns = make_feature_cols(), model_dir = OUTDIR)

tf.logging.set_verbosity(tf.logging.INFO)

model.train(input_fn = make_train_input_fn(df_train, num_epochs = None), steps=2600)

print_rmse(model, df_valid)

# train_model(model)
# run_test_predict(model)

# df_predict = pd.read_csv('./datasets/sales_to_predict.csv', header = 0, names = CSV_COLUMNS)

# print("============Future predict============")

predict(model, df_test)

data = {
    # "quantidadetotal": [np.nan],
    "produto": [3936],
    "ano": [2019],
    "mes": [3],
    "crescimento": [30.0]
}
df = pd.DataFrame.from_dict(data)

print("============Future 2 predict============")

predict(model, df)
