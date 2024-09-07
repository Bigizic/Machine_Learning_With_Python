## Softmax Regression

`` Focus on how to convert a linear classifier into a multi-class classifier, including multinomial logistic regression or softmax regression ``

This has to do with multi-class classification

``SoftMax regression is similar to logistic regression, and the softmax function converts the actual distances, that is, dot products of  𝑥
  with each of the parameters  𝜃𝑖
  for the  𝐾
  classes. This is converted to probabilities using the following:``

```bash
𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑥,𝑖)=𝑒𝜃𝑇𝑖𝐱∑𝐾𝑗=1𝑒𝜃𝑇𝑗𝑥

$softmax(x,i) = \frac{e^{ \theta_i^T \bf x}}{\sum_{j=1}^K e^{\theta_j^T x}} $
```

``The training procedure is almost identical to logistic regression. Consider the three-class example where  𝑦∈{0,1,2}
  we would like to classify  𝑥1
 . We can use the softmax function to generate a probability of how likely the sample belongs to each class:``


```bash
[𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑥1,0),𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑥1,1),𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑥1,2)]=[0.97,0.2,0.1]

$[softmax(x_1,0),softmax(x_1,1),softmax(x_1,2)]=[0.97,0.2,0.1]$
```


``The index of each probability is the same as the class. We can make a prediction using the argmax function:``


```bash
𝑦̂ =𝑎𝑟𝑔𝑚𝑎𝑥𝑖{𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑥,𝑖)}

$\hat{y}=argmax_i  \{softmax(x,i)\}$
```

``For the previous example, we can make a prediction as follows:
``


```bash
𝑦̂ =𝑎𝑟𝑔𝑚𝑎𝑥𝑖{[0.97,0.2,0.1]}=0

$\hat{y}=argmax_i  \{[0.97,0.2,0.1]\}=0$
```


``
The sklearn does this automatically, but we can verify the prediction step, as we fit the model:``

```py
lr = LogisticRegression(random_state=0).fit(X, y)
```


``We generate the probability using the method predict_proba:``

```py
probability=lr.predict_proba(X)
```

``We can plot the probability of belonging to each class; each column is the probability of belonging to a class and the row number is the sample number.``


```py
plot_probability_array(X,probability)
```


``Here, is the output for the first sample:``


```py
probability[0,:]
```


``We see it sums to one.``


```py
probability[0,:].sum()
```


``We can apply the  𝑎𝑟𝑔𝑚𝑎𝑥 function.``


```py
np.argmax(probability[0,:])
```


``We can apply the  𝑎𝑟𝑔𝑚𝑎𝑥 function to each sample.``


```py
softmax_prediction=np.argmax(probability,axis=1)
softmax_prediction
```


``We can verify that sklearn does this under the hood by comparing it to the output of the method  predict  .``


```py
yhat =lr.predict(X)
accuracy_score(yhat,softmax_prediction)
```
