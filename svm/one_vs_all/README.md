# Continuation of SoftMax Regression for Multi-Class Classification <a href="https://github.com/Bigizic/Machine_Learning_With_Python/tree/main/regression/softmax_regression">here</a>

## One vs All is used for SVMS

## One vs. All (One-vs-Rest)


``For one-vs-all classification, if we have K classes, we use K two-class classifier models. The number of class labels present in the dataset is equal to the number of generated classifiers. First, we create an artificial class we will call this "dummy" class. For each classifier, we split the data into two classes. We take the class samples we would like to classify, the rest of the samples will be labelled as a dummy class. We repeat the process for each class. To make a classification, we use the classifier with the highest probability, disregarding the dummy class.``

### Train Each Classifier

``Here, we train three classifiers and place them in the list my_models. For each class we take the class samples we would like to classify, and the rest will be labelled as a dummy class. We repeat the process for each class. For each classifier, we plot the decision regions. The class we are interested in is in red, and the dummy class is in blue. Similarly, the class samples are marked in blue, and the dummy samples are marked with a black x.``


```py
#dummy class
dummy_class=y.max()+1
#list used for classifiers 
my_models=[]
#iterate through each class
for class_ in np.unique(y):
    #select the index of our  class
    select=(y==class_)
    temp_y=np.zeros(y.shape)
    #class, we are trying to classify 
    temp_y[y==class_]=class_
    #set other samples  to a dummy class 
    temp_y[y!=class_]=dummy_class
    #Train model and add to list 
    model=SVC(kernel='linear', gamma=.5, probability=True)    
    my_models.append(model.fit(X,temp_y))
    #plot decision boundary 
    decision_boundary (X,temp_y,model,iris)
```


``For each sample we calculate the probability of belonging to each class, not including the dummy class.``


```py
probability_array=np.zeros((X.shape[0],3))
for j,model in enumerate(my_models):
    real_class=np.where(np.array(model.classes_)!=3)[0]
    probability_array[:,j]=model.predict_proba(X)[:,real_class][:,0]
```

``Here, is the probability of belonging to each class for the first sample.``

```py
probability_array[0,:]
```
```bash
array([9.92070795e-01, 1.05488542e-01, 4.41392470e-12])
```


``As each is the probability of belonging to the actual class and not the dummy class, it does not sum to one.``


```py
probability_array[0,:].sum()
```
```bash
1.0975593362547043
```


``We can plot the probability of belonging to the class. The row number is the sample number.``

```py
plot_probability_array(X,probability_array)
```

``We can apply the  𝑎𝑟𝑔𝑚𝑎𝑥 function to each sample to find the class.``

```py
one_vs_all=np.argmax(probability_array,axis=1)
one_vs_all
```
```bash
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```


``We can calculate the accuracy.``


```py
accuracy_score(y,one_vs_all)
```
```bash
0.9466666666666667
```


``We see the accuracy is less than the one obtained by sklearn, and this is because for SVM, sklearn uses one vs one; let's verify it by comparing the outputs.``


```py
accuracy_score(one_vs_all,yhat)
```
```bash
0.9733333333333334
```

``We see that the outputs are different, now lets implement one vs one.``
