import pandas as pd, pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('features.csv')
X = df.drop(columns=['commit_hash','label'])
y = df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
clf = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
os.makedirs('models',exist_ok=True)
pickle.dump(clf,open('models/codedna_model.pkl','wb'))
print('Model saved!')
print('Accuracy:',round((y_pred==y_test).mean()*100,1),'%')