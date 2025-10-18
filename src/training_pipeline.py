import joblib
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data, preprocess_data, train_model, test_model, load_model, save_model


data = load_data('data/train.csv')
x_train, y_train, x_val, y_val = preprocess_data(df=data, target='loan_status', mode='train')

rfc = RandomForestClassifier(random_state=42)
hgbc = HistGradientBoostingClassifier(random_state=42)
knc = KNeighborsClassifier()
label_encoder = load_model('models/label_encoder.joblib')

fitted_rfc = train_model(rfc, x_train, y_train)
fitted_hgbc = train_model(hgbc, x_train, y_train)

y_pred = test_model(fitted_rfc, x_val)
print(y_pred.shape)

print(label_encoder.inverse_transform(y_pred.reshape(-1, 1)[0]))
print(label_encoder.inverse_transform(y_val.values.reshape(-1, 1)[0]))

save_model(fitted_rfc, 'models/trained_RFC.joblib')
save_model(fitted_hgbc, 'models/trained_HGBC.joblib')
save_model(fitted_hgbc, 'models/trained_KNC.joblib')

