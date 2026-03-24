## Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib


##1 Data Input
data = pd.read_csv('bank-full.csv',sep=';')
print(data.head())
print(data.shape)
print(data.dtypes)
data.isnull().sum()


##2  convert output
data['y']=data['y'].map({'yes':1,'no':0})


##3 Categorical and numerical colums
categorical = data.select_dtypes(include=['object','str']).columns.tolist()
print("categorical = ",categorical)
numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numerical = ", numerical)


##4 Get categorical values
for i in categorical:
    print(i)
    print(data[i].unique())
    print(data[i].value_counts())
    print('\n')


##5 Convert Categorical 

#job
rare=['entrepreneur','self-employed','unemployed','housemaid','student']
data['job_grouped']=data['job'].replace(rare,'other')
data=data.drop(columns=['job'])

# education
tuk={'secondary':2,'tertiary':3,'primary':1,'unknown':0}
data['education_rank']=data['education'].map(tuk)
data=data.drop(columns=['education'])

# default
data['default']=data['default'].map({'yes':1,'no':0})

# housing
data['housing']=data['housing'].map({'yes':1,'no':0})

# loan
data['loan']=data['loan'].map({'yes':1,'no':0})

# month
mr = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
cal=data['month'].map(mr)
data['month_sin']=np.sin(2*np.pi*cal/12)
data['month_cos']=np.cos(2*np.pi*cal/12)
data = data.drop(columns=['month'])

# Hot Encoding
data = pd.get_dummies(data, columns=['job_grouped','marital','contact','poutcome'], drop_first=True)


##6 Converting Numerical

print(numerical)

# outliers
outlier_counts = {}
for col in numerical:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[(data[col] < lower) | (data[col] > upper)]
    outlier_counts[col] = len(outliers)

df = pd.DataFrame(list(outlier_counts.items()), columns=['Column', 'Outlier_Count'])
print(df)

# outlier capping
data['y'].value_counts()
majority = data[data['y'] == 0]
minority = data[data['y'] == 1]

for col in numerical:
    Q1 = majority[col].quantile(0.25)
    Q3 = majority[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    majority[col] = np.where(majority[col] < lower, lower, majority[col])
    majority[col] = np.where(majority[col] > upper, upper, majority[col])

data = pd.concat([majority, minority])
    
# checking for - for log
print("campaign min:", data['campaign'].min())
print("previous min:", data['previous'].min())
print("Duration min:", data['duration'].min())
print("Balance min:", data['balance'].min())

# Balance signed log
data['balance_log'] = np.sign(data['balance']) * np.log1p(np.abs(data['balance']))

# Duration normal log
data['duration_log'] = np.log1p(data['duration'])
print("balance_log skew:", data['balance_log'].skew())
print("duration_log skew:", data['duration_log'].skew())
data=data.drop(columns=['balance','duration'])

# campaign and previous
data['campaign'] = np.log1p(data['campaign'])
data['previous'] = np.log1p(data['previous'])

# Handling Pday For -1
data['previous_contact'] = (data['pdays'] != -1).astype(int)
data['pdays'] = data['pdays'].replace(-1, 0)
data['pdays'] = np.log1p(data['pdays'])


##7 checking
print(data.head())


##8 Correlation
corr = data.corr(numeric_only=True)
corr_target = corr['y'].sort_values(ascending=False)
print(corr_target)


##9 Drop
data=data.drop(columns=['default','poutcome_other','marital_married','day'])


##10 Data Table
def is_binary(col):
    return set(col.unique()) <= {0, 1}

summary = []
binary_col = []
numerical_col = []
categorical_col = []

for col in data.columns:
    dtype = data[col].dtype
    if dtype == 'object':
        col_type = 'Categorical'
        binary = False
        categorical_col.append(col)
    else:  # numeric
        if is_binary(data[col]):
            col_type = 'Binary'
            binary = True
            binary_col.append(col)
        else:
            col_type = 'Numerical'
            binary = False
            numerical_col.append(col)
    summary.append((col, col_type, binary))
    
summary_df = pd.DataFrame(summary, columns=['Column', 'Type', 'Is_Binary'])
type_order = {'Binary': 0, 'Numerical': 1, 'Categorical': 2}
summary_df['Sort_Order'] = summary_df['Type'].map(type_order)
summary_df = summary_df.sort_values(['Sort_Order', 'Column']).drop(columns=['Sort_Order']).reset_index(drop=True)
print(summary_df)


##11 Prepare For Split
x = data.drop('y', axis=1)
y = data['y']


##12 Split For Train,Test,Val
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
x_test, x_val, y_test, y_val = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


##13 Scaling
sc=StandardScaler()
x_train[numerical_col] = sc.fit_transform(x_train[numerical_col])
x_test[numerical_col] = sc.transform(x_test[numerical_col])
x_val[numerical_col] = sc.transform(x_val[numerical_col])


##14 Oversampling
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)


##15 Model
model = LogisticRegression(
    solver='lbfgs',
    C=0.5,
    max_iter=1000
)


##16 Train
model.fit(x_train, y_train)


##17 Best F1
y_probs = model.predict_proba(x_val)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print("Best Threshold:", best_threshold)
print("Best F1:", f1_scores[best_idx])


##18 Metrice And Result
y_pred = (model.predict_proba(x_test)[:,1] > best_threshold).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


##19 Saveing the model
joblib.dump(model, 'banker_best_model_th_0_7982930846443228.pkl')

