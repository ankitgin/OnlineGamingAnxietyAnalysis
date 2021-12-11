df["GAD_T"] = df["GAD_T"]
bins = [-1, 5, 10, 15, 21]
names = [0, 1, 2, 3]

df['GAD_T'] = pd.cut(df['GAD_T'], bins, labels=names)

cols = ['Hours', 'streams', 'Age', 'Employed', 'Unemployed', 'school_student', 'university_student',
       'Female', 'Male', 'Other', 'all', 'fun', 'improve', 'others_whyplay',
       'relax', 'win', 'All', 'Multiplayer-Acquaintances',
       'Multiplayer-Friends', 'Multiplayer-Strangers', 'Others_playstyle',
       'SinglePlayer', 'earn_little', 'earn_living', 'no_earning','SPIN_T','SWL_T']
out_cols = ["GAD_T"]
num_cols = len(cols)

X = df[cols]
y = df[out_cols]

cols = ['Hours', 'streams', 'Age', 'Female', 'Male', 'Other','Multiplayer-Acquaintances', 'Multiplayer-Friends', 'Multiplayer-Strangers', 'Others_playstyle', 'SinglePlayer', 'SPIN_T', 'SWL_T']

X_train, X_test, y_train, y_test = train_test_split(df[cols], y, stratify=y)

scaler_x = MinMaxScaler()
scaler_x.fit(X_train)
xscale = scaler_x.transform(X_train)
xscale_test = scaler_x.transform(X_test)
best = 0
clf = MLPClassifier(hidden_layer_sizes = (20,10),max_iter=500,activation='relu',solver='adam',learning_rate='adaptive',random_state=42).fit(xscale, y_train)
score = clf.score(xscale_test, y_test)
print("score: ", score)

clf.get_params()
