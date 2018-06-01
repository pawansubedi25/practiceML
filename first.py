from sklearn import tree, datasets

iris = datasets.load_iris() #default iris dataset class provided with sklearn package, other is digits

iris_features = iris.data[:-1]
iris_targetClass = iris.target[:-1]

test_feature = iris.data[-1]
test_answer = iris.target[-1]

#[height, weight, shoe size]
X = [[181,80,44], [171,70,443],[185,82,45],[171,75,44],[141,50,34],[178,83,45], [189,80,44], [178,76,42],[181,82,41],[174,78,41],[161,70,44],[181,76,45]]
Y = ['male','female','male','female','female','male','male','female','male','female','female','male']
#target_set = set(iris_targetClass)

print((iris_features))
#print(test_feature)

#print(target_set)

print(len(iris_features), len(iris_targetClass))

clf = tree.DecisionTreeClassifier() #prediction model

clf = clf.fit(iris_features,iris_targetClass) #training

prediction = clf.predict(test_feature) #prediction

print(prediction)

print(test_answer - prediction)

