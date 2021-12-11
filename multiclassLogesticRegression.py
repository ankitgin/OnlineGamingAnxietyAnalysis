from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

##Grouping 0-21 to 4 classes
d = {range(0, 5): 0, range(5, 10): 1, range(10, 15): 2, range(15,22): 3}
df_orig['groupedGAD'] = df['GAD_T'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

def logisticReg(cols):
  # print(cols)
  score = 0
  ##run 20 times to normalize error
  for i in range(20): 
    out_cols = ["groupedGAD"]
    num_cols = len(cols)
    X = df_orig[cols]
    y = df_orig[out_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


    pca = PCA(n_components=int(0.3*num_cols)+1)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)


    scaler_x_train = MinMaxScaler()
    scaler_x_train.fit(X_train)
    train_X = scaler_x_train.transform(X_train)
    train_y = y_train

    scaler_x_test = MinMaxScaler()
    scaler_x_test.fit(X_test)
    test_X = scaler_x_test.transform(X_test)
    test_y = y_test

    model = LogisticRegression(multi_class='multinomial', max_iter = 300) 
    clf = model.fit(train_X, train_y)
    pred = clf.predict(test_X)
    score += clf.score(test_X,test_y)
  print("\nLogistic Regression score:",score/20)
  
  
  from sklearn.ensemble import AdaBoostClassifier

##Adaboost on Multiclass logistic regression
def adaboost(cols):
  out_cols = ["groupedGAD"]
  num_cols = len(cols)
  X = df_orig[cols]
  y = df_orig[out_cols]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
  
  scaler_x_train = MinMaxScaler()
  scaler_x_train.fit(X_train)
  train_X = scaler_x_train.transform(X_train)
  train_y = y_train

  scaler_x_test = MinMaxScaler()
  scaler_x_test.fit(X_test)
  test_X = scaler_x_test.transform(X_test)
  test_y = y_test

  model = LogisticRegression(multi_class='multinomial', max_iter = 300)
  classifier = AdaBoostClassifier(base_estimator=model, n_estimators=50, learning_rate=0.01)
  classifier = classifier.fit(train_X, train_y)
  score = classifier.score(test_X,test_y)
  print("Adaboost score: ", score)
  
  #All
cols = ["Age","Hours", "all" , "fun",  "improve", "relax" ,"others_whyplay", "win","All", "Multiplayer-Acquaintances","Multiplayer-Friends", "Multiplayer-Strangers", "Others_playstyle", "SinglePlayer", "earn_little", "no_earning","earn_living"]
logisticReg(cols)
adaboost(cols)
