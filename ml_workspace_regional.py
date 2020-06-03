import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from osgeo import gdal
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import KFold, cross_validate
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from xgboost import XGBClassifier#, XGBRFClassifier
import shap

shap.initjs()
pd.plotting.register_matplotlib_converters()
sns.set(rc={'figure.figsize':(12,12), "font.size":9,"axes.titlesize":9,
            "axes.labelsize":9})   
# %matplotlib inline
scoring = ['balanced_accuracy', 'f1', 'precision', 'recall']
#%%
dataf1 = pd.read_csv('/Users/adibanpa/Dropbox/ML/engineered_data_7.csv')
dataf1 = dataf1.drop(axis=1, columns='Unnamed: 0')
dataf2 = pd.read_csv('/Users/adibanpa/Dropbox/ML/engineered_data_7.csv')
dataf2 = dataf2.drop(axis=1, columns='Unnamed: 0')

dataf1 = dataf1.drop(columns = ['deep10', 'deep10_mean', 'deep10_min', 'deep10_max', 
                                'deep10_var' ,'deep10_med', 'deep10_std', 'deep10_kurt', 
                                'deep10_skew', 'deep2', 'deep2_mean', 'deep2_min', 
                                'deep2_max', 'deep2_var', 'deep2_med', 'deep2_std', 
                                'deep2_kurt', 'deep2_skew', 'deep4', 'deep4_mean', 
                                'deep4_min', 'deep4_max', 'deep4_var', 'deep4_med',
                                'deep4_std','deep4_kurt', 'deep4_skew','deep6', 
                                'deep6_mean', 'deep6_min','deep6_max', 'deep6_var', 
                                'deep6_med', 'deep6_std','deep6_kurt', 'deep6_skew',])

dataf2 = dataf2.drop(columns = ['deep10', 'deep10_mean', 'deep10_min', 'deep10_max', 
                                'deep10_var' ,'deep10_med', 'deep10_std', 'deep10_kurt', 
                                'deep10_skew', 'deep2', 'deep2_mean', 'deep2_min', 
                                'deep2_max', 'deep2_var', 'deep2_med', 'deep2_std', 
                                'deep2_kurt', 'deep2_skew', 'deep4', 'deep4_mean', 
                                'deep4_min', 'deep4_max', 'deep4_var', 'deep4_med',
                                'deep4_std','deep4_kurt', 'deep4_skew','deep6', 
                                'deep6_mean', 'deep6_min','deep6_max', 'deep6_var', 
                                'deep6_med', 'deep6_std','deep6_kurt', 'deep6_skew',])

ds = gdal.Open(r'C:\Users\adibanpa\CloudStation\Reproj_layers\au_reproj.tif')

#%%
# Plot distribution of gold        
f_au, ax_au = plt.subplots(figsize=(7, 7))
ax_au.set(yscale="log")
sns.distplot(dataf1['au'], kde = False, color="blue")
ax_au.set(xlabel='Number of Au showing/occurence', ylabel='log (number of pixels)')
#%%
to_train = list(dataf2.keys())
to_train.remove('au')
to_train.remove('au_mean')
to_train.remove('au_max')
to_train.remove('au_min')
to_train.remove('au_var')
to_train.remove('au_med')
to_train.remove('au_std')
to_train.remove('au_skew')
to_train.remove('au_kurt')
#%%
xpos = to_train.index('x')
ypos = to_train.index('y')
#%%
dataf2['au'][dataf1['au']>2.] = 1
dataf2['au'][dataf1['au']<=2.] = 0

#%%
###Scaling that data using a minmax scaler between 0 and 1.
X = dataf2[to_train].values
y = dataf2['au'].values
#scaler = MinMaxScaler()
#scaler.fit(X)
#X_scl = scaler.transform(X)
#y_scl = (y-np.min(y))/np.ptp(y)
#%%
###Sampling 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
rus = RandomUnderSampler(random_state=0)
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_sample(X, y) #oversampling
X_rus, y_rus = rus.fit_sample(X, y) #undersampling

#X_ros, y_ros = X,y #oversampling
#X_rus, y_rus = X,y #undersampling
#%%
#Indices for the testing set defined based on pixel position
ind = np.where(np.logical_and(X.T[xpos].astype(int)>=50, X.T[xpos].astype(int)<=100))
indos = np.where(np.logical_and(X_ros.T[xpos].astype(int)>=50, X_ros.T[xpos].astype(int)<=100))
indus = np.where(np.logical_and(X_rus.T[xpos].astype(int)>=50, X_rus.T[xpos].astype(int)<=100))
X_test_ = X[ind]
X_train_ = X[~np.in1d(np.arange(len(X.T[xpos])), ind)]
X_train_os_ = X_ros[~np.in1d(np.arange(len(X_ros.T[xpos])), indos)]
X_train_us_ = X_rus[~np.in1d(np.arange(len(X_rus.T[xpos])), indus)]
#sorting indices because the pixels are moved around during sampling
usind = np.lexsort((X_train_us_.T[1], X_train_us_.T[0]))
osind = np.lexsort((X_train_os_.T[1], X_train_os_.T[0]))
X_trainus_ = X_train_us_[usind]
X_trainos_ = X_train_os_[osind]
y_test = y[ind]
y_train = y[~np.in1d(np.arange(len(X.T[xpos])), ind)]
y_trainos = y_ros[~np.in1d(np.arange(len(X_ros.T[xpos])), indos)]
y_trainus = y_rus[~np.in1d(np.arange(len(X_rus.T[xpos])), indus)]
y_trainus = y_trainus[usind]
y_trainos = y_trainos[osind]
#%%
# x and y position removed from featyres matrix
X_test = np.delete(X_test_, [xpos, ypos], axis=1)
X_trainos = np.delete(X_trainos_, [xpos, ypos], axis=1)
X_trainus = np.delete(X_trainus_, [xpos, ypos], axis=1)
X_train = np.delete(X_train_, [xpos, ypos], axis=1)
#%%
geox = np.arange(ds.GetGeoTransform()[0], ds.GetGeoTransform()[0]+max(dataf1['x'])*2000, 2000)
geoy = np.arange(ds.GetGeoTransform()[3], ds.GetGeoTransform()[3]+max(dataf1['y'])*2000, 2000)
plt.figure()
plt.plot(ds.GetGeoTransform()[0] + (X_test_.T[0] * 2000), ds.GetGeoTransform()[3] - (X_test_.T[1] * 2000), 'bo', label='Testing set')
plt.plot(ds.GetGeoTransform()[0] + (X_train_.T[0] * 2000), ds.GetGeoTransform()[3] - (X_train_.T[1] * 2000), 'ro', label='Training set')
#plt.xticks(geox)
#plt.yticks(geoy)
plt.legend()
#plt.gca().invert_yaxis()
#%%
###XGBOOST Modelling on oversampled data
xgb1 = XGBClassifier(verbosity=0,  booster="gbtree",
                     eval_metric = "logloss",
                     eval_set = [(X_test, y_test)],
                     early_stopping_rounds = 10,
                     colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,
                     max_depth = 3,
                     n_estimators=500, 
                     learning_rate=0.001,
                     min_child_weight=100,
                     gamma=90, 
                     subsample=0.5,
                     reg_alpha=400,
                     reg_lambda = 400,
                     random_state=12, n_jobs=-1)
#%%
#cross validation of training data
kf = KFold(n_splits=5)
scores= cross_validate(estimator=xgb1, X=X_trainos, y=y_trainos, scoring=scoring, cv=kf)
print("balanced accuracy: {}".format(np.mean(scores['test_balanced_accuracy'])))
print("f1: {}".format(np.mean(scores['test_f1'])))
print("Recall: {}".format(np.mean(scores['test_recall'])))
print("Precision: {}".format(np.mean(scores['test_precision'])))
#%%
xgb1.fit(X_trainos, y_trainos)
#%%
y_probos = xgb1.predict_proba(X_test).T[1]
y_predos = xgb1.predict(X_test)
y_predos2 = y_probos
#%%
y_predos2[y_probos >= 0.55] = 1
y_predos2[y_probos < 0.55] = 0
#%%
y_true = y_test
print("XGBoost: Oversampled")
print("Accuracy on training set: {:.3f}".format(xgb1.score(X_trainos, y_trainos))) 
print("Accuracy on test set: {:.3f}".format(xgb1.score(X_test, y_test)))
cfos = confusion_matrix(y_true, y_predos2)
print('tn, fp, fn, tp')
cf_normalize = (cfos-np.min(cfos))/np.ptp(cfos)
tn, fp, fn, tp = cfos.ravel()
print(cfos.ravel())
print("balanced accuracy: {}".format(balanced_accuracy_score(y_true, y_predos2)))
print("f1: {}".format(f1_score(y_true, y_predos2, average='binary')))
print("Recall: {}".format(recall_score(y_true, y_predos2)))
print("Precision: {}".format(precision_score(y_true, y_predos2)))
#%%    
## Undersampling
xgbus = XGBClassifier(verbosity=0,  booster="gbtree",
                     eval_metric = "logloss",
                     eval_set = [(X_test, y_test)],
                     early_stopping_rounds = 10,
                     colsample_bytree=0.5, colsample_bylevel=0.5, colsample_bynode=0.5,
                     max_depth = 3,
                     n_estimators=400, 
                     learning_rate=0.001,
                     min_child_weight=45,
                     gamma=15, 
                     subsample=0.5,
                     reg_alpha=60,
                     reg_lambda = 60,
                     random_state=12, n_jobs=-1)
#%%
kf = KFold(n_splits=5)
scores= cross_validate(estimator=xgbus, X=X_trainus, y=y_trainus, scoring=scoring, cv=kf)
print("balanced accuracy: {}".format(np.mean(scores['test_balanced_accuracy'])))
print("f1: {}".format(np.mean(scores['test_f1'])))
print("Recall: {}".format(np.mean(scores['test_recall'])))
print("Precision: {}".format(np.mean(scores['test_precision'])))
#%%
xgbus.fit(X_trainus, y_trainus)
#%%
y_probus = xgbus.predict_proba(X_test).T[1]
y_predus = xgbus.predict(X_test)
y_predus2 = y_probus
#%%
y_predus2[y_probus >= 0.54] = 1
y_predus2[y_probus < 0.54] = 0
#%%
y_true = y_test
print("XGBoost: undersampled")
print("Accuracy on training set: {:.3f}".format(xgbus.score(X_trainus, y_trainus))) 
print("Accuracy on test set: {:.3f}".format(xgbus.score(X_test, y_test)))
cfus = confusion_matrix(y_true, y_predus2)
print('tn, fp, fn, tp')
cf_normalize1 = (cfus-np.min(cfus))/np.ptp(cfus)
tn, fp, fn, tp = cfus.ravel()
print(cfus.ravel())
print("balanced accuracy: {}".format(balanced_accuracy_score(y_true, y_predus2)))
print("f1: {}".format(f1_score(y_true, y_predus2, average='binary')))
print("Recall: {}".format(recall_score(y_true, y_predus2)))
print("Precision: {}".format(precision_score(y_true, y_predus2)))
#%%
## No sampling 
#xgb1 = XGBClassifier(verbosity=0, 
#                     scale_pos_weight = 7,
#                     max_depth = 3,
#                     n_estimators=200, 
#                     learning_rate=0.01, 
#                     gamma=10, 
#                     subsample=0.5,
#                     reg_alpha=5,
#                     reg_lambda = 5,
#                     random_state=12, n_jobs=-1)
#xgb1.fit(X_train, y_train)
#y_prob = xgb1.predict_proba(X_test).T[1]
#y_pred = xgb1.predict(X_test)
#y_true = y_test
#print("XGB")
#print("Accuracy on training set: {:.3f}".format(xgb1.score(X_train, y_train))) 
#print("Accuracy on test set: {:.3f}".format(xgb1.score(X_test, y_test)))
#cf = confusion_matrix(y_true, y_pred)
#print('tn, fp, fn, tp', cf)
#cf_normalize = (cf-np.min(cf))/np.ptp(cf)
#tn, fp, fn, tp = cf.ravel()
#print(cf.ravel())
#print("balanced accuracy: {}".format(balanced_accuracy_score(y_true, y_pred)))
#print("f1: {}".format(f1_score(y_true, y_pred, average='binary')))
#print("Recall: {}".format(recall_score(y_true, y_pred)))
#print("Precision: {}".format(precision_score(y_true, y_pred)))
#%%
#Random forest classifier for oversampling
forest1 = RandomForestClassifier(max_depth = 3, 
                      n_estimators=200, max_features = len(to_train)-3,
                      min_samples_split = 0.1,
                      random_state=12, n_jobs=-1)
#%%
kf = KFold(n_splits=5)
scores= cross_validate(estimator=forest1, X=X_trainos, y=y_trainos, scoring=scoring, cv=kf)
print("balanced accuracy: {}".format(np.mean(scores['test_balanced_accuracy'])))
print("f1: {}".format(np.mean(scores['test_f1'])))
print("Recall: {}".format(np.mean(scores['test_recall'])))
print("Precision: {}".format(np.mean(scores['test_precision'])))
#%%
forest1.fit(X_trainos, y_trainos)
#%%
y_probrf1 = forest1.predict_proba(X_test).T[1]
y_predrf1 = forest1.predict(X_test)
#%%
y_predrf1_ = y_probrf1
y_predrf1_[y_probrf1 >= 0.55] = 1
y_predrf1_[y_probrf1 < 0.55] = 0
#%%
print("RF")
print("Accuracy on training set: {:.3f}".format(forest1.score(X_trainos, y_trainos))) 
print("Accuracy on test set: {:.3f}".format(forest1.score(X_test, y_test)))
cf2 = confusion_matrix(y_true, y_predrf1)
print('tn, fp, fn, tp', cf2)
cf_normalize2 = (cf2-np.min(cf2))/np.ptp(cf2)
tnrf1, fprf1, fnrf1, tprf1 = cf2.ravel()
print(cf2.ravel())
print("balanced accuracy: {}".format(balanced_accuracy_score(y_true, y_predrf1_)))
print("f1: {}".format(f1_score(y_true, y_predrf1_, average='binary')))
print("Recall: {}".format(recall_score(y_true, y_predrf1_)))
print("Precision: {}".format(precision_score(y_true, y_predrf1_)))
#%%
##Random Forest classifier for undersampling
forest2 = RandomForestClassifier(max_depth = 3, 
                      n_estimators=200, max_features = len(to_train)-3,
                      min_samples_split = 0.1,
                      random_state=12, n_jobs=-1)
#%%
kf = KFold(n_splits=5)
scores= cross_validate(estimator=forest2, X=X_trainus, y=y_trainus, scoring=scoring, cv=kf)
print("balanced accuracy: {}".format(np.mean(scores['test_balanced_accuracy'])))
print("f1: {}".format(np.mean(scores['test_f1'])))
print("Recall: {}".format(np.mean(scores['test_recall'])))
print("Precision: {}".format(np.mean(scores['test_precision'])))
#%%
forest2.fit(X_trainus, y_trainus)
#%%
y_probrf2 = forest2.predict_proba(X_test).T[1]
y_predrf2 = forest2.predict(X_test)
y_predrf2_ = y_probrf2
#%%
y_predrf2_[y_probrf2 >= 0.54] = 1
y_predrf2_[y_probrf2 < 0.54] = 0
#%%
print("RF")
print("Accuracy on training set: {:.3f}".format(forest2.score(X_trainus, y_trainus))) 
print("Accuracy on test set: {:.3f}".format(forest2.score(X_test, y_test)))
cf3 = confusion_matrix(y_true, y_predrf2)
print('tn, fp, fn, tp', cf3)
cf_normalize2 = (cf3-np.min(cf3))/np.ptp(cf3)
tnrf2, fprf2, fnrf2, tprf2 = cf3.ravel()
print(cf3.ravel())
print("balanced accuracy: {}".format(balanced_accuracy_score(y_true, y_predrf2_)))
print("f1: {}".format(f1_score(y_true, y_predrf2_, average='binary')))
print("Recall: {}".format(recall_score(y_true, y_predrf2_)))
print("Precision: {}".format(precision_score(y_true, y_predrf2_)))
#%%

##Random Forest classifier for unsampled data
#forest3 = RandomForestClassifier(max_depth = 5, 
#                      n_estimators=1000, 
#                      random_state=12, n_jobs=-1)
#forest3.fit(X_train, y_train)
#y_probrf3 = forest3.predict_proba(X_test).T[1]
#y_predrf3 = forest3.predict(X_test)
#print("RF")
#print("Accuracy on training set: {:.3f}".format(forest3.score(X_train, y_train))) 
#print("Accuracy on test set: {:.3f}".format(forest3.score(X_test, y_test)))
#cf4 = confusion_matrix(y_true, y_predrf3)
#print('tn, fp, fn, tp', cf4)
#cf_normalize3 = (cf4-np.min(cf4))/np.ptp(cf4)
#tnrf3, fprf3, fnrf3, tprf3 = cf4.ravel()
#print(cf4.ravel())
#print("Accuracy: {}".format((tprf3+tnrf3)/len(y_test)))
#print("Recall: {}".format(tprf3/(tprf3+fnrf3)))
#print("Precision: {}".format(tprf3/(tprf3+fprf3)))
#%%
# ---------SHAP---------
to_train2 = np.delete(to_train, [xpos, ypos], axis=0)
# DF, based on which importance is checked
X_importance = X_test
#%%
# Explain model predictions using shap library:
explainer = shap.TreeExplainer(xgb1)
shap_values = explainer.shap_values(X_importance)
#%%
plt.figure()
shap.summary_plot(shap_values, X_importance, max_display = 20, 
                  feature_names = to_train, show=False)
plt.tight_layout()
plt.show()
#plt.savefig('summaryplot.png', dpi=200)
#%%
shap.dependence_plot('ewdeep10_kurt', shap_values, 
                     X_importance, interaction_index=None,
                     feature_names=to_train2, x_jitter=1, alpha=0.4, dot_size=20
                    )
#%%
explainer2 = shap.TreeExplainer(xgbus)
shap_values2 = explainer2.shap_values(X_importance)
#%%
plt.figure()
shap.summary_plot(shap_values2, X_importance, max_display = 20, 
                  feature_names = to_train2, show=False)
plt.tight_layout()
plt.show()
#plt.savefig('summaryplot.png', dpi=200)
#%%
for i in to_train[3:74]:
    shap.dependence_plot(i, shap_values2, 
                         X_importance, interaction_index=None, alpha=0.4,
                         feature_names=to_train2, x_jitter=1, dot_size=20)

#%% Random forest dependence plot
#depplotrf1 = plot_partial_dependence(forest1, X_trainos, to_train[3:74], feature_names = to_train2)
#%%
fig = plt.figure(figsize=(15,10))
#fig.suptitle('Prediction maps compared to map of ore deposits/showings', fontsize=28)
normalizedau = (dataf1['au'][ind[0]]-np.min(dataf1['au'][ind[0]]))/np.ptp(dataf1['au'][ind[0]])
ax1 = fig.add_subplot(121)
ax1.set_title('Map of deposits/showings', fontsize=16)
sc1 = ax1.scatter(X_test_.T[xpos], X_test_.T[ypos], marker = "s", 
                  c = y_test , cmap='gnuplot')
ax1.set_ylabel('Pixel', fontsize=14)
ax1.set_xlabel('Pixel', fontsize=14)
fig.gca().invert_yaxis()
plt.colorbar(sc1)

ax2 = fig.add_subplot(122)
ax2.set_title('XGBoost: Prediction map', fontsize=16)
sc2 = ax2.scatter(X_test_.T[xpos], X_test_.T[ypos], marker = "s",
                  c = y_predos2 , cmap='gnuplot')
#ax2.set_ylabel('Pixel', fontsize=14)
ax2.set_xlabel('Pixel', fontsize=14)
fig.gca().invert_yaxis()
plt.colorbar(sc2)

#ax3 = fig.add_subplot(1,3,3)
#ax3.set_title('Random Forest: Prediction map', fontsize=16)
#sc3 = ax3.scatter(X_test_0.T[xpos], X_test_0.T[ypos],  marker = "s", 
#                  c = y_predrf2, cmap='gnuplot')
##ax3.set_ylabel('Pixel', fontsize=14)
#ax3.set_xlabel('Pixel', fontsize=14)
#fig.gca().invert_yaxis()
#plt.colorbar(sc3)

#plt.savefig('goldpred.png')
plt.show()

#%%
aupred = np.full((176,224), np.nan)
yyy = y_predrf2_.astype(bool)
aupred[X_test_.T[0].astype(int), X_test_.T[1].astype(int)] = yyy
aupred = aupred.T
dst_filename = '/Users/adibanpa/Dropbox/ML/rfus.tif'
x_pixels = aupred.shape[1]  # number of pixels in x
y_pixels = aupred.shape[0]  # number of pixels in y
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(dst_filename,x_pixels, y_pixels, 1,gdal.GDT_Byte)
dataset.GetRasterBand(1).WriteArray(aupred)

geotrans=ds.GetGeoTransform()  #get GeoTranform from existed 'data0'
proj=ds.GetProjection() #you can get from a exsited tif or import 
dataset.SetGeoTransform(geotrans)
dataset.SetProjection(proj)
dataset.FlushCache()

#%%
#fig = plt.figure()
#ax = fig.add_subplot(111) 
#ax.scatter(dataf2['x'].values, dataf2['y'].values, c=dataf2['au'].values)
#plt.show()

#%%
plt.figure()
plt.scatter(x=dataf2['x'], y=dataf2['y'], c=dataf2['nsdeep2'])

