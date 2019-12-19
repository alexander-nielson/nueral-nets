R Notebook
================

This is a simple R program ot build, train, and test a regression NN on the Boston MASS data set.

First, load the necessary packages

``` r
library("neuralnet")
```

    ## Warning: package 'neuralnet' was built under R version 3.5.3

``` r
library(MASS)
```

    ## Warning: package 'MASS' was built under R version 3.5.3

Set the seed for reproducibility.

``` r
set.seed(1)
```

Load the data and scale it

``` r
data = Boston
max_data <- apply(data, 2, max)
min_data <- apply(data, 2, min)
data_scaled <- scale(data,center = min_data, scale = max_data - min_data)
```

split the data using a 70-30 split

``` r
index = sample(1:nrow(data),round(0.70*nrow(data)))
train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])
```

To save time and avoid having to write out all the variables, we get a names vector, and then create a formula. inside the formula so that medv is the response. we then select all the other 'names'/predictors in the names vector and put a '+' so that we can fully construct the "response ~ var1+var2 +..." format

``` r
n = names(data)
f = as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
```

Create a neural net

``` r
net_data = neuralnet(f,data=train_data,hidden=10,linear.output=T)
```

plot the neural net

``` r
plot(net_data)
```

next we are interested to see how well or NN performs. To do this, we first get the predicted medb of testing data using the neural network.

``` r
predict_net_test <- compute(net_data,test_data[,1:13])
```

``` r
predict_net_test_start <- predict_net_test$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
```

``` r
test_start <- as.data.frame((test_data$medv)*(max(data$medv)-min(data$medv))+min(data$medv))
```

``` r
MSE.net_data <- sum((test_start - predict_net_test_start)^2)/nrow(test_start)
```

``` r
Regression_Model <- lm(medv~., data=data)
summary(Regression_Model)
```

    ## 
    ## Call:
    ## lm(formula = medv ~ ., data = data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -15.595  -2.730  -0.518   1.777  26.199 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  3.646e+01  5.103e+00   7.144 3.28e-12 ***
    ## crim        -1.080e-01  3.286e-02  -3.287 0.001087 ** 
    ## zn           4.642e-02  1.373e-02   3.382 0.000778 ***
    ## indus        2.056e-02  6.150e-02   0.334 0.738288    
    ## chas         2.687e+00  8.616e-01   3.118 0.001925 ** 
    ## nox         -1.777e+01  3.820e+00  -4.651 4.25e-06 ***
    ## rm           3.810e+00  4.179e-01   9.116  < 2e-16 ***
    ## age          6.922e-04  1.321e-02   0.052 0.958229    
    ## dis         -1.476e+00  1.995e-01  -7.398 6.01e-13 ***
    ## rad          3.060e-01  6.635e-02   4.613 5.07e-06 ***
    ## tax         -1.233e-02  3.760e-03  -3.280 0.001112 ** 
    ## ptratio     -9.527e-01  1.308e-01  -7.283 1.31e-12 ***
    ## black        9.312e-03  2.686e-03   3.467 0.000573 ***
    ## lstat       -5.248e-01  5.072e-02 -10.347  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4.745 on 492 degrees of freedom
    ## Multiple R-squared:  0.7406, Adjusted R-squared:  0.7338 
    ## F-statistic: 108.1 on 13 and 492 DF,  p-value: < 2.2e-16

``` r
test <- data[-index,]
predict_lm <- predict(Regression_Model,test)
MSE.lm <- sum((predict_lm - test$medv)^2)/nrow(test)
MSE.net_data
```

    ## [1] 12.06928

``` r
MSE.lm
```

    ## [1] 26.99266
