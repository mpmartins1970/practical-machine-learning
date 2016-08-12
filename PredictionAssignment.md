# Prediction Assignment Writeup
mpmartins1970  
2016/08/12  

********************************************************************************

## Executive Summary

In this report, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants are used to quantify how well they did each particular activity. The goal of this report is to predict the manner in which they did the exercise, describing how the model was built, the expected out of sample error and why the choices were made. The predict model is used to predict 20 different test cases. Using exploratory data analysis and machine learning theory, the findings showed that Random Forest has the best accuracy for this testing dataset.

## Exploratory Data Analysis

Loading the training and test datasets that previously have been downloaded to working directory in local machine. The datasets have 160 variables.


```r
# Libraries
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
# Load the training and testing dataset
trainingDS <- read.csv("./pml-training.csv", na.strings = c("NA","#DIV/0!"))
testingDS <- read.csv("./pml-testing.csv", na.strings = c("NA","#DIV/0!"))
```

Removing all the variables containing NA values, the near zero variance (NVZ) variables and id/timestamp variables.


```r
# Cleaning data
trainingDS <- trainingDS[,colSums(is.na(trainingDS)) == 0]
trainingDS <- trainingDS[,-nearZeroVar(trainingDS)]
trainingDS <- trainingDS[,-c(1:7)]
```

Splitting data into a 70% training data set and a 30% testing data set to estimate the out of sample error of the predictor.


```r
# Splitting data
inTrain <- createDataPartition(y = trainingDS$classe, p = 0.70,list = F)
training <- trainingDS[inTrain,] 
testing <- trainingDS[-inTrain,] 
```

## Prediction Model Building

The problem to be resolved is a classification one, so, using random forest, the out of sample error should be small. Random forest is used for the training dataset using cross-validation. 


```r
# Making report reproducible
set.seed(1970)

# Fitting Random Forest model
trc5 <- trainControl(method = "cv", number = 5, allowParallel = TRUE, verbose = TRUE)
modelRF <- train(classe~.,data = training, method = "rf", trControl = trc5, verbose = FALSE)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=26 
## - Fold1: mtry=26 
## + Fold1: mtry=51 
## - Fold1: mtry=51 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=26 
## - Fold2: mtry=26 
## + Fold2: mtry=51 
## - Fold2: mtry=51 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=26 
## - Fold3: mtry=26 
## + Fold3: mtry=51 
## - Fold3: mtry=51 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=26 
## - Fold4: mtry=26 
## + Fold4: mtry=51 
## - Fold4: mtry=51 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=26 
## - Fold5: mtry=26 
## + Fold5: mtry=51 
## - Fold5: mtry=51 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

```r
modelRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, verbose = FALSE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.74%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    1    1    0    1 0.0007680492
## B   14 2631   13    0    0 0.0101580135
## C    1   16 2377    2    0 0.0079298831
## D    0    0   47 2204    1 0.0213143872
## E    0    0    1    4 2520 0.0019801980
```

In sequence, the fitted model generated is examined with the testing sample from the partitioned training dataset to evaluate the accuracy and estimated error of prediction.


```r
# Predicting with the Random Forest Model
predictionRF <- predict(modelRF, testing)
confMatrixRF <- confusionMatrix(predictionRF, testing$classe)
confMatrixRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674   10    0    0    0
##          B    0 1126    3    0    0
##          C    0    3 1022   18    0
##          D    0    0    1  942    2
##          E    0    0    0    4 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9886   0.9961   0.9772   0.9982
## Specificity            0.9976   0.9994   0.9957   0.9994   0.9992
## Pos Pred Value         0.9941   0.9973   0.9799   0.9968   0.9963
## Neg Pred Value         1.0000   0.9973   0.9992   0.9955   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1913   0.1737   0.1601   0.1835
## Detection Prevalence   0.2862   0.1918   0.1772   0.1606   0.1842
## Balanced Accuracy      0.9988   0.9940   0.9959   0.9883   0.9987
```

The accuracy of the modeling method above is 99,4% with a small out of sample error. As a consequence, it could be expected almost all of the submitted test cases will be correct.

## Predicting with the Testing Data

Applying the Random Forest model to predict the 20 test cases provided (testing dataset) as shown below.


```r
# Predicting using pml-testing.csv data
predictTestingDS <- predict(modelRF, newdata = testingDS)
predictTestingDS
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusions

From these data, after validation in the prediction quizz, the goal was accomplished since all the submitted test cases were correct.


********************************************************************************
