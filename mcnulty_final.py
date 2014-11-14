# Dara Elass
# Project Mcnulty

# import modules
import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale

# ###########################################################################
# VARIABLES:
    # YEAR (Survey year)
    # SERIAL (Sequential Serial Number, Household Record)
    # STRATA (Stratum for variance estimation)
    # PSU (Primary sampling unit (PSU) for variance estimation)
    # HHWEIGHT (Household weight, final annual)
    # PERNUM (Person number within family (from reformatting))
    # PERWEIGHT (Final basic annual weight)
    # SAMPWEIGHT (Sample Person Weight)
    # ASTATFLG (Sample adult flag)
    # CSTATFLG (Sample child flag)
    # ACNEYR (Had severe acne, past 12months)
    # ANEMIAYR (Had anemia, past 12 months)
    # ANXFREQYR (Frequently anxious, past 12 months)
    # ANXIOUSYR (Had anxiety or stress, past 12 months)
    # AREFLUXYR (Had acid reflux or heartburn, past 12 months)
    # BACKPAINYR (Had back pain, past 12 months)
    # BOWELCONYR (Had severe constipation, past 12 months)
    # BOWELINFYR (Had inflammatory bowel disease, past 12 months)
    # CHOLHIGHYR (Had high cholesterol, past 12 months)
    # DEPFREQYR (Frequently depressed, past 12 months)
    # FATIGUEYR (Had fatigue, past 12 months)
    # YTQYOGYR (Practiced yoga, past 12 months)
    # YTQTAIYR (Practiced Tai chi, past 12 months)
    # YTQIGYR (Practiced Qi gong, past 12 months)
    # HERYR (Took herbal supplements, past 12 months)
    # VITANY (Took any vitamin or mineral supplements, past 12 months)

    # 1 = NO
    # 2 = YES
# ###########################################################################

# set up dataframe
character_widths = [4,6,4,3,6,2,9,9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # define field lengths
headers = ['year','serial','strata','psu','hhweight','pernum','perweight','sampweight','astatflg','cstatflg','acneyr','anemiayr','anxfreqyr','anxiousyr','arefluxyr','backpainyr','bowelconyr','bowelinfyr','cholhighyr','depfreqyr','fatigueyr','ytqyogyr','ytqtaiyr','ytqigyr','heryr','vitany'] # define headers
data = pd.read_fwf('/Users/dmelass/Projects/mcnulty/mcnulty_hd_adepr/dara/health.dat',widths = character_widths, header = None) # put data in df
data.columns = headers # assign headers
data = data.drop([col for col in ['acneyr','anemiayr','anxfreqyr','anxiousyr','arefluxyr','backpainyr','bowelconyr','bowelinfyr','depfreqyr','fatigueyr','serial','strata','psu','hhweight','pernum','perweight','sampweight','astatflg','cstatflg']],1) # drop columns not needed

current_data = data
for col in ['cholhighyr','ytqyogyr','vitany']:
    current_data = current_data[current_data[col].isin([1,2])] # only want values 1 and 2

# set up training/test set
X = current_data[['ytqyogyr','vitany']]
Y = current_data[['cholhighyr']]
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=.15, random_state=5)

# KNN model (accuracy ~0.63, recall values 0.28/0.73
neigh_best = KNeighborsClassifier(n_neighbors=4)
neigh_best.fit(X_train, Y_train)
y_predicted = neigh_best.predict(X_test)
print classification_report(Y_test, y_predicted)
print neigh_best.get_params()


print '\nALL DATA:'
print 'data entries:', len(current_data)
print 'did yoga in the past year - no', len(X[X['ytqyogyr']==1])
print 'did yoga in the past year - yes', len(X[X['ytqyogyr']==2])
print 'took vitamins/supplements in the past year - no', len(X[X['vitany']==1])
print 'took vitamins/supplements in the past year - yes', len(X[X['vitany']==2])
print 'high cholesterol - no:', len(Y[Y['cholhighyr']==1])
print 'high cholesterol - yes:', len(Y[Y['cholhighyr']==2])

X_new = X
X_new['ytqyogyr'] = X_new['ytqyogyr']-1
X_new['vitany'] = X_new['vitany']-1
X_new.to_csv('vizdata.csv', index=False,header = ['yoga','vitamins'])
print 'created csv for visualization'

print '\nTRAINING SET:'
print 'data entries', len(X_train)
print 'did yoga in the past year - no', (np.transpose(X_train)[0]==1).sum()
print 'did yoga in the past year - yes', (np.transpose(X_train)[0]==2).sum()
print 'took vitamins/supplements in the past year - no', (np.transpose(X_train)[1]==1).sum()
print 'took vitamins/supplements in the past year - yes', (np.transpose(X_train)[1]==2).sum()
print 'high cholesterol - no:', (np.transpose(Y_train)[0]==1).sum()
print 'high cholesterol - yes:', (np.transpose(Y_train)[0]==2).sum()

print '\nTEST SET:'
print 'data entries', len(X_test)
print 'did yoga in the past year - no', (np.transpose(X_test)[0]==1).sum()
print 'did yoga in the past year - yes', (np.transpose(X_test)[0]==2).sum()
print 'took vitamins/supplements in the past year - no', (np.transpose(X_test)[1]==1).sum()
print 'took vitamins/supplements in the past year - yes', (np.transpose(X_test)[1]==2).sum()
print 'high cholesterol - no:', (np.transpose(Y_test)[0]==1).sum()
print 'high cholesterol - yes:', (np.transpose(Y_test)[0]==2).sum()

# Logistic regression
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
y_predicted_log = model_logistic.predict(X_test)
score_log = accuracy_score(Y_test,y_predicted_log)
print 'Logistic regression accuracy score: %.4f' % score_log
print classification_report(Y_test, y_predicted_log)

# ###########################################################################

# VARIABLES:
   #9/10 = chol: serum cholestoral in mg/dl - high is 240; borderline high is 200-239
   #36/37 = trestbpd: resting blood pressure
   #15/16: fbs: fasting blood sugar

# set up dataframe
print 'working on old data'
data = pd.read_csv('/Users/dmelass/Projects/mcnulty/mcnulty_hd_adepr/dara/all_data.csv', header = None)
data = data[[9,36,15]]
headers = ['cholest','restingbp','fastingbloodsugar']
data.columns = headers

# remove rows with bad data ('/N')
data = data[-(data['restingbp']=='\N')]
data = data[-(data['fastingbloodsugar']=='\N')]
data = data[-(data['cholest']=='\N')]
print 'number of data points after removal:', len(data)

cholestvalues = data.cholest.values
for i in range(len(cholestvalues)):
    if int(cholestvalues[i]) >= 240:
        print cholestvalues[i]
        cholestvalues[i] = 1
    else:
        cholestvalues[i] = 0

# set up training/test sets
X = data[['restingbp','fastingbloodsugar']]
Y = data['cholest']
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=.3, random_state=5)

# KNN model (accuracy ~0.63, recall values 0.28/0.73
neigh_best = KNeighborsClassifier(n_neighbors=4)
neigh_best.fit(X_train, Y_train)
y_predicted = neigh_best.predict(X_test)
print classification_report(Y_test, y_predicted)
print neigh_best.get_params()

# Logistic regression
model_logistic = LogisticRegression()
model_logistic.fit(X_train,Y_train)
y_predicted_log = model_logistic.predict(X_test)
score_log = accuracy_score(Y_test,y_predicted_log)
print 'Logistic regression accuracy score: %.4f' % score_log
print classification_report(Y_test, y_predicted_log)
