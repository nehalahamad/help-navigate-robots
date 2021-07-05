from flask import Flask, render_template, request
# from werkzeug import secure_filename
import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
import pickle
# from sklearn.preprocessing.data

def resultant(x,y,z=0,w=0):
    return (x**2 + y**2 + z**2 + w**2)**0.5

def quaternion_to_euler_angle(q0, q1, q2, q3):
    a = 2*(q0*q1 + q2*q3)
    b = 1 - 2*(q1*q1 + q2*q2)
    t1 = np.arctan2(a,b)
    
    t2 = 2*(q0*q2 - q3*q1)
    
    c = 2*(q0*q3 + q1*q2)
    d = 1 - 2*(q2*q2 + q3*q3)
    t3 = np.arctan2(c,d)
    return pd.concat((t1, t2, t3), axis=1)

from scipy.stats import entropy
def Entropy(x):
    return entropy(x.value_counts()/len(x))
    
def feature_engg(data_X):
    data_X = data_X.drop(['row_id', 'measurement_number'], axis=1)
    
    data_X.fillna(0,inplace=True)
    data_X.replace(-np.inf,0,inplace=True)
    data_X.replace(np.inf,0,inplace=True)
    
    data_X["resultant_angular_velocity"] = resultant(data_X["angular_velocity_X"], data_X["angular_velocity_Y"],\
                                                     data_X["angular_velocity_Z"])
    data_X["resultant_linear_acceleration"] = resultant(data_X["linear_acceleration_X"], data_X["linear_acceleration_Y"],\
                                                        data_X["linear_acceleration_Z"])
    
    data_X[["euler_t1", "euler_t2", "euler_t3"]] = quaternion_to_euler_angle(data_X['orientation_W'], data_X['orientation_X'],\
                                                                             data_X['orientation_Y'], data_X['orientation_Z'])

    new_df = pd.DataFrame()

    funct1 = {'min':'min()','max':'max()','mean':'mean()','std':'std()', 'median':'median()', 'mad':'mad()', \
              'quantile25':'quantile(0.25)', 'quantile75':'quantile(0.75)', 'skewness':'skew()'}
    
    for op in funct1:
        df = eval('data_X.groupby(["series_id"]).{}'.format(funct1[op]))
        df.columns = df.columns.map(lambda x: x+'_'+op)
        new_df = pd.concat([new_df, df], axis=1)
        
    for col in data_X.columns:
        if col not in ['row_id', 'series_id', 'measurement_number']:
            new_df[col + '_kurtosis'] = data_X.groupby('series_id')[col].apply(lambda x: x.kurtosis())
            new_df[col + '_abs_max'] = data_X.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))
            new_df[col + '_abs_min'] = data_X.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))            
            new_df[col + '_energy'] = data_X.groupby('series_id')[col].apply(lambda x: sum(x**2)/len(x))
            new_df[col + '_mean_abs_change'] = data_X.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
            new_df[col + '_mean_change_of_abs_change'] = data_X.groupby('series_id')[col].apply(lambda x: \
                                                                                        np.mean(np.diff(np.abs(np.diff(x)))))
            new_df[col + '_max_to_min'] = new_df[col + '_max'] / new_df[col + '_min']
    
    return new_df

def final(X, y=None):
    '''final() takes two values, source and target, if only one value(source) will be given then it will
    return only predicted output, if two value(source, target) will be given then it will return accuracy'''
    
    # featurizing the data
    new_df = feature_engg(X)
    # scaling the features
    new_df_scaled = sk.transform(new_df)
    
    # predicting the output 
    X_pred = lgbm.predict(new_df_scaled)

    accuracy = 'to get accuracy upload source and target data together'
   #  return X_pred, accuracy
    if y is None:
        return X_pred, accuracy
    else:
        return X_pred, metrics.accuracy_score(y['surface'].values, X_pred)
# =======================================================

app = Flask(__name__)

# importing pretrained model and scaler
sk = pickle.load(open('flask_TutorialByKrish\scaler.pkl', 'rb'))
lgbm = pickle.load(open('flask_TutorialByKrish\lgbm.pkl', 'rb'))

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      s = request.files['sourcefile']
      X = pd.read_csv(s)

      try:
         t = request.files['targetfile']
         y = pd.read_csv(t)
      except:
         y = None

      output, accu = final(X, y)

      if y is None:
          output1 = list(zip(' '*len(output), output))
      else:
          output1 = list(zip(y['surface'].values, output))

      return render_template('upload.html', result=output1, accuracy=accu)

if __name__ == '__main__':
   app.run(debug = True)
