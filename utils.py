import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def getX(df : pd.DataFrame) -> pd.DataFrame:
    df_new = df.filter(regex = "cell")
    return df_new

def feature_engineering(data : pd.DataFrame, y : np.array, lags : int, mean_var = True):
    data_length = data.shape[0]
    data_new = data.loc[lags:data_length]
    y_new = y[lags:]
    var_names = data.columns
    # Create lags colums
    for i in range(1,lags+1):
        data_lag = data.loc[(lags-i):(data_length-i-1)]
        data_lag = data_lag.add_suffix(f"_lag_{i}")
        
        data_new.reset_index(drop=True, inplace=True)
        data_lag.reset_index(drop=True, inplace=True)
        data_new = pd.concat([data_new, data_lag], axis = 1)
    
    # Creat lag means and variance
    if mean_var:
        for i in range(len(var_names)):
            df_var = data_new.filter(regex = var_names[i])
            data_new[str(var_names[i]+"_mean")]  = df_var.mean(axis = 1) 
            data_new[str(var_names[i]+"_var")] = df_var.var(axis = 1)
    
    return data_new, y_new


def select_non_missing(X : pd.DataFrame, y : pd.DataFrame):
    missing = np.isnan(y)
    y_pure = y[~missing]
    X_pure = X.loc[np.array(~missing)]
    
    return X_pure, y_pure


def pca_decomp(X_train, X_val, n_comp = 20):
    pca = PCA(n_components = n_comp)
    
    rename = {}
    for name in X_train.columns:
        rename[name] = "pca_" +str(name)
    
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_val = pd.DataFrame(pca.transform(X_val))

    X_train = X_train.rename(columns = rename)
    X_val = X_val.rename(columns = rename)
    
    explained_variance = pca.explained_variance_ratio_
    print(sum(explained_variance))
    return X_train, X_val


def to_radiant(cos, sin):
    cos = np.clip(cos, -1, 1)
    sin = np.clip(sin, -1, 1)
    
    radiant = np.arccos(cos)
    radiant = radiant + (sin<0)*np.pi
    return radiant

def to_multi(y):
    y_sin = np.sin(y)
    y_cos = np.cos(y)
    y_multi =np.array([y_cos, y_sin]).transpose()
    return y_multi
    