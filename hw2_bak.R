---
title: "HW2, ISyE 6501"
author: "Hank Igoe"
date: "1/24/2022"
output: pdf_document
---
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
rm(list=ls())
```
```{r message=F,results='hide'}
# R packages used:
library(purrr); library(magrittr);library(assertthat)
library(kernlab);library(zeallot);
library(rlist); library(readr); library(kknn)
# Global vars for convenience:
c(KMAX, KCV, CUTOFF_1, CUTOFF_2) %<-% c(2,2, 520, 580) # 3.1.1: c(50,5,520,580)
CS <- 10^(-4:0) # C=1, .1 etc
```
## Question 3.1

>Using the same data set (credit_card_data.txt or credit_card_data-headers.txt) as in Question 2.2, use the ksvm or kknn function to find a good classifier:
 
 >a. using cross-validation (do this for the k-nearest-neighbors model; SVM is optional); and
 b. splitting the data into training, validation, and test data sets (pick either KNN or SVM; the other is optional).

### Question 3.1.a.i: Cross-validation of $k$-nearest neighbors model.

First, we will read in our credit card data:
```{r preprocess}
responseCol <- 11
colLength <- 11 # in case need 12th column of original row nums 
set.seed(1)
options(digits=3)
c(cutoff1, cutoff2) %<-% c(CUTOFF_1, CUTOFF_2) # for partitioning to train/validate/test
ccMatrix <- read_tsv("../data/credit_card_data-headers.txt", show_col_types=F)

# helper functions:
scaleExcept <- function(df, cols_) {
  nCols <- ncol(df)
  ns <- names(df)
  for (i in 1:nCols)
    if (!(i %in% cols_)) df[, i] <- scale(df[, i])
  names(df) <- ns
  df
}

shuffle <- function(df) {
  nRows <- nrow(df)
  df <- df[sample(nRows), ]
}

# scale, add col of original rows, shuffle
preprocess <- function(df, noScaleCols) {
  df <- scaleExcept(df, noScaleCols)
  df <- cbind(df, i=row.names(df) %>% as.integer)
  df %<>% shuffle
}

ccData <- preprocess(as.data.frame(ccMatrix), c(responseCol))
ccDataTrain <- ccData[1:cutoff1, ][ ,1:colLength] 
ccDataValid <- ccData[(cutoff1 + 1):cutoff2, ][ ,1:colLength]
ccDataTest <- ccData[cutoff2:654, ][ ,1:colLength]
ccDataCV <- rbind(ccDataTrain, ccDataValid)
nRows <- nrow(ccData)
kkNN_kernels <- c("rectangular", "triangular", "epanechnikov", "gaussian", "rank", "optimal")
```

Now for the cross-validation:
```{r}
# Helper functions:
pred_kNN_rowi <- function(data, k_, i, kernel="rectangular", scale=T) {
  model_knn = kknn(as.factor(R1) ~ .,
                   data[-i,],
                   data[ i,],
                   k = k_,
                   kernel=kernel,
                   scale = scale)
  fitted(model_knn) %>% as.integer-1 # factors become 1,2 instead of 0,1
}

test_K <- function(data, k, kernel="rectangular", scale=T) {
  nrows <- nrow(data) # dim(data)[[1]]
  pred_i <- function(i) pred_kNN_rowi(data, k, i)

  predictions <- lapply(1:nrows, pred_i)
  sum(predictions == data[ ,responseCol]) / nrows
}

arg_max <- function(xs) {
  argAcc <- function(i, acc) {
    if (i == 0) return(acc)
    acc <- max(acc, xs[i])
    argAcc(i-1, acc)
  }
  argAcc(length(xs)-1, xs[length(xs)])
}

# Use train.kknn to select a (kernel, kNN) model by cross-validating on
# (k-1) subsets of the data, then validate that (kernel, kNN) setting
# on the kTh remaining part of the data and return the best model
# found so that its quality can be evaluated on a final set of test data.
# Returns 2 lists: first is models, second is their validated quality
train_kknnCV <- function(df, kmax=KMAX, kcv=KCV,
  formula=as.factor(R1)~., kernel=kkNN_kernels, distance=2, ks=NULL, scale=T) {
  nRows <- nrow(df)

  assert_that(nRows %% kcv == 0)
  subsetLen <- nRows %/% kcv
  trainAcc <- function(i, acc, validAcc) {
    if (i > kcv) return(list(acc, validAcc))
    validationRange <- ((i - 1) * subsetLen + 1) : (i*subsetLen)
    validationData <- df[validationRange, ]
    trainData <- df[-validationRange, ]

    x <- train.kknn(distance=distance, ks=ks,
      formula = formula, data = trainData, kmax = kmax,
      kernel = kernel, scale = scale)

    acc[[i]] <- x$best.parameters
    c(kernel_i, kNN_i) %<-% acc[[i]]
    validAcc[[i]] <- test_K(validationData, kernel=kernel_i, k=kNN_i)
    trainAcc(i+1, acc, validAcc)
  }
  trainAcc(1, rep(list(list(0,0)), kcv), rep(list(0), kcv))
}

# Test the model with the highest quality per the results of train_kknnCV:
test_kknnCV <- function(dfCV, dfTest, kmax=KMAX, kcv=KCV,
  formula=as.factor(R1)~., kernel=kkNN_kernels, distance=2, ks=NULL, scale=T) {
  # cvs[[1]] is the list of models, which are (kernel, kNN) pairs
  # cvs[[2]] is quality validation results for each model

  cvs <- train_kknnCV(dfCV, kmax, kcv, formula, kernel, distance, ks, scale)
  print("Trained models:"); print(cvs[[1]])
  print("Validated qualities:"); print(cvs[[2]])
  models <- cvs[[1]]
  validationResults <- cvs[[2]] %>% unlist # all validation results
  modelArgToTest <- arg_max(validationResults %>% unlist) # which to test
  modelToTest <- models[[modelArgToTest]]

  modelQuality <- test_K(dfTest, kernel=modelToTest[1],
    k=modelToTest[2] %>% as.integer, scale)

  print("Model with highest cross-validation quality, its validation quality, and its test quality:")
  print(modelToTest); print(validationResults[[modelArgToTest]]); print(modelQuality)
  list(modelToTest %>% unlist, modelQuality)
}

validatedModel_TestQuality <- test_kknnCV(ccDataCV, ccDataTest, kmax=KMAX, kcv = KCV)
```
We see that the winning (kernel, kNN) model in the cross-validation contest is ("optimal", 17) because its validation quality of 89% is higher than the rest, and our estimate of that model's quality in general is 84% based on the results of running it against test data which it was neither trained nor validated on.

### 3.1.b.i: Training, validation and testing of SVM models (not cross-validated).

In this analysis we will train several SVM models on the training data and measure the quality of each trained model on the validation data. Then we will choose the model which has the highest validation quality and give an unbiased estimate of its quality in general after measuring its effectiveness on the test data.

```{r}
# Run ksvm on df_ with C and return (a, a0, pred), i.e., (weights, intercept, predictions)
C2ModelWeightsIntercept <- function(df_, C_, kernel_="vanilla", modelOnly=T) {
  if (df_ %>% class != "matrix") df_ %<>% as.matrix

  model <- ksvm(df_[,1:10],df_[,11],type="C-svc",kernel=kernel_,C=C_,scaled=T)
  if (modelOnly) return(model)
  # calculate a1…am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])

  # calculate a0
  a0 <- -model@b

  # see what the model predicts
  # pred <- predict(model, df_[,1:10])
  list(model, a, a0) # return model, weights (a) and intercept (a0)
}

pred2successRate <- function(pred, df_, responseCol=responseCol) {
  sum(pred == df_[ ,responseCol]) / nrow(df_) 
}

Cs2TrainedModels <- function(dfTrain, Cs=CS, kernel_="vanilla", modelOnly=T) {
  models <- map(Cs, function(C_) C2ModelWeightsIntercept(dfTrain, C_, kernel_, modelOnly))
  #models <- map(CS, function(C_) C2ModelWeightsIntercept(ccDataTest, C_))
}

Cs2ValidatedModels <- function(dfTrain, dfValidate, Cs=CS, kernel_="vanilla", modelOnly=T) {
  tms <- Cs2TrainedModels(dfTrain, Cs, kernel_, modelOnly)
  N <- length(tms)
  validAcc <- function(i, iAcc, succAcc) {
    if (i > N) return(list(tms, iAcc, succAcc))
    pred_i <- predict(tms[[i]], dfValidate[ ,1:10])
    succ_i <- pred2successRate(pred_i, dfValidate, responseCol)
    if (succ_i > succAcc)
      c(iAcc, succAcc) %<-% c(i, succ_i)
    validAcc(i+1, iAcc, succAcc)
  }
  validAcc(1, 1, 0)
}

bestValidatedModel <- function(
  dfTrain=ccDataTrain, dfValidate=ccDataValid, Cs=CS, kernel_="vanilla") {
  models_i_succ <- Cs2ValidatedModels(dfTrain, dfValidate, Cs, kernel_)
}

# If model_i_succ is present, it is a triple (m, i, succ) from a CV
# Pass model=NULL if using model_i_succ
qualityOfModel <- function(model, models_i_succ=NULL, dfTest=ccDataTest,
  responseCol=responseCol) {
  if (!is.null(models_i_succ)) {
    c(models, i, vsucc) %<-% models_i_succ
    model <- models[[i]]
    print("Models, model index in CV, validation success rate:")
    print(models_i_succ)
  }
  pred <- predict(model, dfTest[, 1:10])
  succ <- pred2successRate(pred, dfTest, responseCol)
  print(c("Test success: ", succ))
  succ
}
```

```{r message=F}
Cs <- CS
print(c("C values:\n", Cs))
vms <- Cs2ValidatedModels(ccDataTrain, ccDataValid, Cs=Cs, kernel="vanilla")
bvmsWithInfo <- bestValidatedModel()
print(c("Validated models:", vms))
qualityOfModel(NULL, bvmsWithInfo, ccDataTest, responseCol)
```
So the second model with the setting of C=0.001 had the highest validation quality with 83%, and then - surprisingly - had a significantly higher test quality of 88%. Randomness could be at work in explaining that, but bear in mind also that an SVM with C=0.001 has a much wider margin, which makes it better at generalizing to data sets on which it was neither trained nor validated, so that quality estimate is not necessarily unrealistic.

### 3.1.2b: Training, validation and testing of $k$-nearest neighbors models

We will train on 70% of the data, validate on 15% and test on 15%.

```{r}
nRows <- nrow(ccData)
trainSize <- as.integer(nRows*.7)
validateSize <- as.integer(nRows*.15)
testSize <- nRows - trainSize - validateSize
paste(trainSize, validateSize, testSize)
set.seed(1)
trainRows <- sample(nRows, trainSize)
validateRows <- (1:nRows)[-trainRows][sample(validateSize+testSize, validateSize)]
testRows <- (1:nRows)[-trainRows][-validateRows]
#paste(lapply(c(trainRows)))
c(trainRows, validateRows, testRows) %>% unique %>% length
#testSample <- vali
```

### Question 4.2 

>The iris data set iris.txt contains 150 data points, each with four predictor variables and one categorical response. The predictors are the width and length of the sepal and petal of flowers and the response is the type of flower. The data is available from the R library datasets and can be accessed with iris once the library is loaded. It is also available at the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Iris ). The response values are only given to see how well a specific method performed and should not be used to build the model.

>Use the R function kmeans to cluster the points as well as possible. Report the best combination of predictors, your suggested value of k, and how well your best clustering predicts flower type.

