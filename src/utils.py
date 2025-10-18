def assert_path(path:str):
    import os
    from pathlib import Path

    cwd = os.getcwd()
    full_path = os.path.join(cwd, path)

    if (os.path.exists(full_path)) and (path.endswith('.csv') or path.endswith('.joblib')):
        return Path(full_path)
    
    else:
        raise ValueError('Invalid file path or file extension')
    
#--------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    try:
        full_data_path = assert_path(data_path)

        import pandas as pd

        df = pd.read_csv(full_data_path, index_col='id')

        return df
    
    except Exception as e:
        raise e

#------------------------------------------------------------------------------------------------------

def split_data(df, target:str='loan_status'):
    try:
        X, Y = df.drop([target], axis=1), df[target]

        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)

        return x_train, x_val, y_train, y_val
    

    except Exception as e:
        raise e
    
#-------------------------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str='loan_status', mode:str='train'):
    try:

        onehot_enc_cols = ['loan_intent', 'person_home_ownership']
        ord_enc_cols = ['loan_grade']
        label_enc_cols =['cb_defaulter_on_file']
        drop_cols = ['person_age', 'cb_person_cred_hist_length', 'person_home_ownership_OTHER']

    
        if mode=='train':
            import pandas as pd
            import numpy as np

            df.rename(columns={'cb_person_default_on_file': 'cb_defaulter_on_file'}, inplace=True)
            df.isnull().sum()

            from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
            ord_enc = OrdinalEncoder()
            lab_enc = LabelEncoder()

            ord_encoded_cols = ord_enc.fit_transform(df[ord_enc_cols])
            onehot_enc_df = pd.get_dummies(df[onehot_enc_cols], dtype=int)

            df = df.join(onehot_enc_df, on='id', how='outer')
            df.drop(onehot_enc_cols, axis=1, inplace=True)

            df['loan_grade'] = ord_encoded_cols
            df['cb_defaulter_on_file'] = lab_enc.fit_transform(np.ravel(df[label_enc_cols]))

            loan_status = df.pop('loan_status')
            df.insert(len(df.columns), 'loan_status', loan_status)

            df.drop(df.filter(regex='loan_intent_').columns, axis=1, inplace=True)
            df.drop(drop_cols, axis=1, inplace=True)

            x_train, x_val, y_train, y_val = split_data(df, target='loan_status')

            from sklearn.preprocessing import QuantileTransformer
            transformer = QuantileTransformer()

            x_train_tfd = transformer.fit_transform(x_train)
            x_val_tfd = transformer.transform(x_val)

            save_model(ord_enc, 'models/ordinal_encoder.joblib')
            save_model(lab_enc, 'models/label_encoder.joblib')
            save_model(transformer, 'models/quantile_transformer.joblib')


            return x_train_tfd, y_train, x_val_tfd, y_val
        

        elif mode=='inference':
            import pandas as pd
            import numpy as np

            ord_encoder = load_model('models/ordinal_encoder.joblib')
            lab_encoder = load_model('models/label_encoder.joblib')
            scaler = load_model('models/quantile_transformer.joblib')

            ord_encoded_cols = ord_encoder.transform(df[ord_enc_cols])
            onehot_enc_df = pd.get_dummies(df[onehot_enc_cols], dtype=int)

            df = df.join(onehot_enc_df, on='id', how='outer')
            df.drop(onehot_enc_cols, axis=1, inplace=True)

            df['loan_grade'] = ord_encoded_cols
            df['cb_defaulter_on_file'] = lab_encoder.transform(np.ravel(df[label_enc_cols]))

            loan_status = df.pop('loan_status')
            df.insert(len(df.columns), 'loan_status', loan_status)

            df.drop(df.filter(regex='loan_intent_').columns, axis=1, inplace=True)
            df.drop(drop_cols, axis=1, inplace=True)

            df_scaled = scaler.transform(df)
            
            return df_scaled


    except Exception as e:
        raise e

#--------------------------------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    try: 
        import joblib

        full_model_path = assert_path(model_path)

        with open(full_model_path, 'rb') as f:
            model = joblib.load(f)

        return model
    

    except Exception as e:
        raise e

#--------------------------------------------------------------------------------------------------------------------------------

def save_model(model, model_path:str):
    try:
        import joblib
        full_model_path = assert_path(model_path)
        
        with open(full_model_path, 'wb') as f:
            joblib.dump(model, f)
    
    except Exception as e:
        raise e

#--------------------------------------------------------------------------------------------------------------------

def train_model(model, x_train, y_train):
    try:
        model.fit(x_train, y_train)

        return model


    except Exception as e:
        raise e
    
#--------------------------------------------------------------------------------------------------------------------

def test_model(model, x_test):
    try:
        preds = model.predict(x_test)

        return preds
    

    except Exception as e:
        raise e
    
#------------------------------------------------------------------------------------------------------------------------


