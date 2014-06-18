Qualitative Activity Recognition of Weight Lifting Exercises
============================================================

Synopsis
--------
Modern devises allow us to collect a large amount of data about personal activity relatively inexpensively. This project uses data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict whether they are performing barbell lifts correctly or incorrectly in 5 different ways. 

Using the Random Forest algorithm, a model was created using 70% of the data for training and 30% for validation. The model produces a high out of sample accuracy of 99% when tested on the validation data set. 


Data Loading and Processing
---------------------------

The [training data set] (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the [test data set] (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) are downloaded from the course website.

We read both data sets.


```r
pmlTrain <- read.csv("pml-training.csv",na.strings=c("NA",""))
pmlTest <- read.csv("pml-testing.csv",na.strings=c("NA",""))

dim(pmlTrain)
```

```
## [1] 19622   160
```

```r
dim(pmlTest)
```

```
## [1]  20 160
```

Load the relevant libraries.


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

Discard non-predictor columns -- x, all timestamps, user-name, new-window, num-window.


```r
rmCols <- grep("X|user_name|window|timestamp",names(pmlTrain))
pmlTrainSub <- pmlTrain[,-rmCols]
dim(pmlTrainSub)
```

```
## [1] 19622   153
```

Calculate proportion of NAs (nulls) for each column. 


```r
nRows <- nrow(pmlTrainSub)
NAcols <- apply(pmlTrainSub, 2, FUN= function(x) sum(is.na(x))/nRows)
```

Discard columns with high NAs (>= 50%).


```r
pmlTrainSub <- pmlTrainSub[,-which(NAcols >= 0.5)]
dim(pmlTrainSub)
```

```
## [1] 19622    53
```

We had trimmed the number of predictor columns to 52 (53 if including the class variable). The columns are:


```r
names(pmlTrainSub)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```



Train the Model
---------------

Create training set using 70% and validation set at 30% from training data to train the model.


```r
pmlTrainIndex <- createDataPartition(y = pmlTrainSub$classe, p=0.7,list=FALSE) 
pmlTrainData <- pmlTrainSub[pmlTrainIndex,]
pmlValData <- pmlTrainSub[-pmlTrainIndex,]

dim(pmlTrainData)
```

```
## [1] 13737    53
```

```r
dim(pmlValData)
```

```
## [1] 5885   53
```

Fit a model using Random Forest. To minimize running time, we set the parameters to run using the "CV" method with 4 folds and also set the function to run in parallel mode. Run-time is approximately 8min.


```r
set.seed(13579)
modFit <- train(classe ~., data = pmlTrainData, method="rf", 
                trControl = trainControl(method = "cv", number = 4),
                allowParallel=T)
```


```r
modFit
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 10302, 10304, 10302, 10303 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.004        0.005   
##   30    1         1      0.003        0.003   
##   50    1         1      0.001        0.001   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Cross validate the prediction function using the validation data set to compute the relevant accuracy measures. 


```r
pred <- predict(modFit, newdata=pmlValData)
confusionMatrix(pred,pmlValData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    5    0    0    0
##          B    0 1133    3    0    0
##          C    0    1 1023   11    1
##          D    0    0    0  953    0
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.995, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.995    0.997    0.989    0.999
## Specificity             0.999    0.999    0.997    1.000    1.000
## Pos Pred Value          0.997    0.997    0.987    1.000    1.000
## Neg Pred Value          1.000    0.999    0.999    0.998    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.162    0.184
## Detection Prevalence    0.285    0.193    0.176    0.162    0.184
## Balanced Accuracy       0.999    0.997    0.997    0.994    1.000
```

The above results show that the out of sample accuracy is very high at **99%**, meaning that the expected out of sample error is **1%**.


Testing on New Data
-------------------

We now apply the model to the unseen test data to predict each record's class. 


```r
predTest <- predict(modFit, newdata=pmlTest)
```

The predictions are:


```r
predTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Create a function to write each prediction into one file for submission to Coursera's Project Submission.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```

We now submit the prediction vector to the function to create the 20 out files in the working directory.


```r
pml_write_files(predTest)
```
