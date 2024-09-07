# Continuation of SoftMax Regression for Multi-Class Classification <a href="https://github.com/Bigizic/Machine_Learning_With_Python/tree/main/regression/softmax_regression">here</a>

## One vs One

``In One-vs-One classification, we split up the data into each class, and then train a two-class classifier on each pair of classes. For example, if we have class 0,1,2, we would train one classifier on the samples that are class 0 and class 1, a second classifier on samples that are of class 0 and class 2, and a final classifier on samples of class 1 and class 2.``


``For  𝐾
  classes, we have to train  𝐾(𝐾−1)/2
  classifiers. So, if  𝐾=3
 , we have  (3𝑥2)/2=3
 classes.``


``To perform classification on a sample, we perform a majority vote and select the class with the most predictions.

Here, we list each class.
``

```py
classes_=set(np.unique(y))
classes_
```
```bash
{0, 1, 2}

```


``Determine the number of classifiers:``


```py
K=len(classes_)
K*(K-1)/2
```
```bash
3.0
```

``We then train a two-class classifier on each pair of classes. We plot the different training points for each of the two classes.``


```py
pairs=[]
left_overs=classes_.copy()
#list used for classifiers 
my_models=[]
#iterate through each class
for class_ in classes_:
    #remove class we have seen before 
    left_overs.remove(class_)
    #the second class in the pair
    for second_class in left_overs:
        pairs.append(str(class_)+' and '+str(second_class))
        print("class {} vs class {} ".format(class_,second_class) )
        temp_y=np.zeros(y.shape)
        #find classes in pair 
        select=np.logical_or(y==class_ , y==second_class)
        #train model 
        model=SVC(kernel='linear', gamma=.5, probability=True)  
        model.fit(X[select,:],y[select])
        my_models.append(model)
        #Plot decision boundary for each pair and corresponding Training samples. 
        decision_boundary (X[select,:],y[select],model,iris,two=True)
```

```py
print(pairs)
```
```bash
['0 and 1', '0 and 2', '1 and 2']

```

``As we can see, our data is left-skewed, containing more "5" star reviews.

Here, we are plotting the distribution of text length.``


```py
pairs
majority_vote_array=np.zeros((X.shape[0],3))
majority_vote_dict={}
for j,(model,pair) in enumerate(zip(my_models,pairs)):

    majority_vote_dict[pair]=model.predict(X)
    majority_vote_array[:,j]=model.predict(X)
```


``In the following table, each column is the output of a classifier for each pair of classes and the output is the prediction:``


```py
pd.DataFrame(majority_vote_dict).head(10)
```
```bash

0 and 1	0 and 2	1 and 2
0	0	0	1
1	0	0	1
2	0	0	1
3	0	0	1
4	0	0	1
5	0	0	1
6	0	0	1
7	0	0	1
8	0	0	1
9	0	0	1
```



``To perform classification on a sample, we perform a majority vote, that is, select the class with the most predictions. We repeat the process for each sample.``


```py
one_vs_one=np.array([np.bincount(sample.astype(int)).argmax() for sample  in majority_vote_array]) 
one_vs_one
```
```bash

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
       2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

``We calculate the accuracy:``


```py
accuracy_score(y,one_vs_one)
```
```bash
0.96
```

``if we complete it to ``sklearn`` it's the same
``

```py
accuracy_score(yhat, one_vs_one)
```
```bash
1.0
```
