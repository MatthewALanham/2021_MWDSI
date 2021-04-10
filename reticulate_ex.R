################################################################################
# Reticulate examples
#
################################################################################

# 1) to set up you python environment
#    https://support.rstudio.com/hc/en-us/articles/360023654474-Installing-and-Configuring-Python-with-RStudio

# 2) install and load library
#install.packages("reticulate")
library(reticulate)

# 3) To configure reticulate to point to the Python executable in your virtualenv
#Sys.setenv(RETICULATE_PYTHON = "python/bin/python")
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Matthew A. Lanham/Desktop/MWDSI/python/Scripts")

# 4) You'll need to restart your R session for the setting to take effect. You 
#    can verify that reticulate is configured for the correct version of Python 
#    using the following command in your R console:
reticulate::py_config()

# your working directory
getwd()

################################################################################
# Flipping back and forth from R and Python
################################################################################
# Currently you should see a single ">" prompt in the console. This means RStudio
# is expecting R code.

# let's construct a 2x2 matrix from a vector of 4 elements
(x <- matrix(1:4, nrow=2, ncol=2))

# Now say you want to change from R to Python. After running this you will notice
# you have a ">>>" prompt in your console. This means RStudio is expecting python
# code. Also notice your Environment window changes.
repl_python()

# Run some python code directly in the console. Notice when you install python
# packages you will see those in your Environment
# y = 5
# print(y)
# import pandas as pd

# To return to the R terminal where you have one ">" in your prompt, just type
# exit in the console
exit

################################################################################
# Calling python files
################################################################################
# call a python file simple_plot.py
reticulate::source_python('C:/Users/Matthew A. Lanham/Desktop/MWDSI/simple_plot.py')
# call a python file test.py
reticulate::source_python('C:/Users/Matthew A. Lanham/Desktop/MWDSI/test.py')


################################################################################
# Predictive Modeling Prototyping with the R caret Package on the adult dataset
################################################################################
# Data source: http://archive.ics.uci.edu/ml/datasets/Adult
# Features:
#age: continuous.
#workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov
#           , State-gov, Without-pay, Never-worked.
#fnlwgt: continuous.
#education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm
#           , Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate
#           , 5th-6th, Preschool.
#education-num: continuous.
#marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed
#               , Married-spouse-absent, Married-AF-spouse.
#occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial
#            , Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical
#            , Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv
#            , Armed-Forces.
#relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#sex: Female, Male.
#capital-gain: continuous.
#capital-loss: continuous.
#hours-per-week: continuous.
#native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany
#               , Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China
#               , Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam
#               , Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador
#               , Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland
#               , Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong
#               , Holand-Netherlands.
#income: >50K, <=50K.
################################################################################
# Option A: Load data from the web
myUrl <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
d <- read.table(file=myUrl, header=F, sep=",", quote="",
                colClasses=c("numeric","factor","numeric","factor","numeric"
                             ,rep("factor",5),rep("numeric",3),rep("factor",2)))
# specify column names
names(d) <- c("age","workclass","fnlwgt","education","educationnum",
              "maritalstatus","occupation","relationship","race","sex",
              "capitalgain","capitalloss","hoursperweek","nativecountry",
              "income")
rm(myUrl)

# Option B: Load data locally
d <- read.table(file="adult.csv", header=T, sep="|")

# examine data structure
str(d)
d$age <- as.numeric(d$age)
d$workclass <- as.factor(d$workclass)
d$fnlwgt <- as.numeric(d$fnlwgt)
d$education <- as.factor(d$education)
d$educationnum <- as.numeric(d$educationnum)
d$maritalstatus <- as.factor(d$maritalstatus)
d$occupation <- as.factor(d$occupation)
d$relationship <- as.factor(d$relationship)
d$race <- as.factor(d$race)
d$sex <- as.factor(d$sex)
d$capitalgain <- as.numeric(d$capitalgain)
d$capitalloss <- as.numeric(d$capitalloss)
d$hoursperweek <- as.numeric(d$hoursperweek)
d$nativecountry <- as.factor(d$nativecountry)
d$income <- as.factor(d$income)

################################################################################
# EDA
################################################################################
options(scipen=999) # remove scientific notation in plots
# set graphics parameters
par(mfrow=c(2,3), bg="white", fg="black",cex.lab=1.2, cex.axis=1.2, cex.main=1.5
    ,las=1, mar=c(4, 3, 2, 1))
# plot numerical features
hist(d$age, main="age", xlab="age", col="gold")
hist(d$capitalgain, main="capitalgain", xlab="capitalgain", col="gold")
hist(d$capitalloss, main="capitalloss", xlab="capitalloss", col="gold")
hist(d$hoursperweek, main="hoursperweek", xlab="hoursperweek", col="gold")
hist(d$fnlwgt, main="fnlwgt", xlab="fnlwgt", col="gold")
hist(d$educationnum, main="educationnum", xlab="educationnum", col="gold")

# graphic
par(mfrow=c(2,2), bg="white", fg="black",cex.lab=1.2, cex.axis=.9, cex.main=1.5
    ,las=3, mar=c(10, 3.5, 2, 1))
plot(d$maritalstatus, main="maritalstatus", col="gold")
plot(d$workclass, main="workclass", col="gold")
plot(d$education, main="education", col="gold")
plot(d$occupation, main="occupation", col="gold")

# graphic
plot(d$relationship, main="relationship", col="gold")
plot(d$race, main="race", col="gold")
plot(d$sex, main="sex", col="gold")
plot(d$nativecountry, main="nativecountry", col="gold")

# graphic
par(mfrow=c(1,1), bg="white", fg="black",cex.lab=1.2, cex.axis=.9, cex.main=1.5
    ,las=3, mar=c(5, 3, 2, 1))
plot(d$income, main="income", col="gold")
# percent less than 50k per year
round(table(d$income)[[1]]/(table(d$income)[[1]]+table(d$income)[[2]]),2)
################################################################################
# Data cleaning
################################################################################
# collapse some of the categories by giving them the same factor level
levels(d$maritalstatus)
levels(d$maritalstatus)[2:4] <- "Married"
levels(d$maritalstatus)

levels(d$workclass)
levels(d$workclass)[c(2,3,8)] <- "Gov"
levels(d$workclass)[c(5,6)] <- "Self"
levels(d$workclass)

# lets delete the factor levels that are labeled as " ?"
d <- d[which(d$workclass != " ?"),]
d <- d[which(d$occupation != " ?"),]
d <- d[which(d$nativecountry != " ?"),]

# in R, the ? levels still exist even though there are 0 records (i.e. counts=0)
table(d$workclass)
table(d$occupation)

# we can easily fix this using the droplevels() function
d$workclass <- droplevels(d$workclass)
d$occupation <- droplevels(d$occupation)
d$nativecountry <- droplevels(d$nativecountry)

# inspect data to see " ?" levels are not permanently removed
table(d$workclass)
table(d$occupation)

# % of rowing having missing values
dim(d[!complete.cases(d),])[[1]]/nrow(d)*100

# Make target variable first column in dataset
d <- d[,c(15,1:14)]
# Make target column name "y"
names(d)[1] <- "y"

str(d)
################################################################################
## Creating Dummy Variables
################################################################################
# Here we want to create a dummy 0/1 variable for every level of a categorical 
# variable
library(caret)
dummies <- dummyVars(y ~ ., data = d)            # create dummies for Xs
ex <- data.frame(predict(dummies, newdata = d))  # actually creates the dummies
names(ex) <- gsub("\\.", "", names(ex))          # removes dots from col names
d <- cbind(d$y, ex)                              # combine target var with Xs
names(d)[1] <- "y"                               # name target var 'y'
rm(dummies, ex)                                  # clean environment
################################################################################
# Identify Correlated Predictors and remove them
################################################################################
# If you build a model that has highly correlated independent variables it can
# lead to unstable models because it will tend to weight those more even though
# they might not be that important

# calculate correlation matrix using Pearson's correlation formula
descrCor <-  cor(d[,2:ncol(d)])                           # correlation matrix
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85) # num Xs with cor > t
summary(descrCor[upper.tri(descrCor)])                    # summarize the cors

# which columns in your correlation matrix have a correlation greater than some
# specified absolute cutoff. Find them and remove them
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
filteredDescr <- d[,2:ncol(d)][,-highlyCorDescr] # remove those specific columns
descrCor2 <- cor(filteredDescr)                  # calculate a new cor matrix
# summarize those correlations to see if all features are now within our range
summary(descrCor2[upper.tri(descrCor2)])

# update dataset by removing those filtered vars that were highly correlated
d <- cbind(d$y, filteredDescr)
names(d)[1] <- "y"

rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)  # clean up
################################################################################
# Identifying linear dependencies and remove them
################################################################################
# Find if any linear combinations exist and which column combos they are.
# Below I add a vector of 1s at the beginning of the dataset. This helps ensure
# the same features are identified and removed.
library(caret)
# first save response
y <- d$y

# create a column of 1s. This will help identify all the right linear combos
d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
names(d)[1] <- "ones"

# identify the columns that are linear combos
comboInfo <- findLinearCombos(d)
comboInfo

# remove columns identified that led to linear combos
d <- d[, -comboInfo$remove]

# remove the "ones" column in the first column
d <- d[, c(2:ncol(d))]

# Add the target variable back to our data.frame
d <- cbind(y, d)

rm(y, comboInfo)  # clean up
################################################################################
# Remove features with limited variation
################################################################################
# remove features where the values they take on is limited
# here we make sure to keep the target variable and only those input
# features with enough variation
nzv <- nearZeroVar(d, saveMetrics = TRUE)
d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])]

################################################################################
# Standardize (and/ normalize) your input features.
################################################################################
# Here we standardize the input features (Xs) using the preProcess() function 
# by performing a min-max normalization (aka "range" in caret).

# Step 1) figures out the means, standard deviations, other parameters, etc. to 
# transform each variable
preProcValues <- preProcess(d[,2:ncol(d)], method = c("range"))
# Step 2) the predict() function actually does the transformation using the 
# parameters identified in the previous step. Weird that it uses predict() to do 
# this, but it does!
d <- predict(preProcValues, d)

################################################################################
# Get the target variable how we want it for modeling with caret
################################################################################
# if greater than 50k make 1 other less than 50k make 0
d$y <- as.factor(ifelse(d$y==" >50K",1,0))
class(d$y)

# make names for target if not already made
levels(d$y) <- make.names(levels(factor(d$y)))
levels(d$y)

# levels of a factor are re-ordered so that the level specified is first and 
# "X1" is what we are predicting. The X before the 1 has nothing to do with the
# X variables. It's just something weird with R. 'X1' is the same as 1 for the Y 
# variable and 'X0' is the same as 0 for the Y variable.
d$y <- relevel(d$y,"X1")

################################################################################
# Data partitioning
################################################################################
set.seed(1234) # set a seed so you can replicate your results
library(caret)

# identify records that will be used in the training set. Here we are doing a
# 70/30 train-test split. You might modify this.
inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .70,   # % of training data you want
                               list = F)
# create your partitions
train <- d[inTrain,]  # training data set
test <- d[-inTrain,]  # test data set

# down-sampled training set
dnTrain <- downSample(x=train[,2:ncol(d)], y=train$y)
names(dnTrain)[91] <- "y"

################################################################################
# Specify cross-validation design
################################################################################
ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=3,        # k number of times to do k-fold
                     classProbs = T,  # if you want probabilities
                     summaryFunction = twoClassSummary, # for classification
                     #summaryFunction = defaultSummary,  # for regression
                     allowParallel=T)

################################################################################
# Train different models
################################################################################
# train a logistic regession on train set 
myModel1 <- train(y ~ .,               # model specification
                  data = train,        # train set used to build model
                  method = "glm",      # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  family = "binomial", # specify the type of glm
                  metric = "ROC"       # performance measure
)
myModel1

# train a logistic regression on down-sampled train set 
myModel2 <- train(y ~ .,               # model specification
                  data = dnTrain,        # train set used to build model
                  method = "glm",      # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  family = "binomial", # specify the type of glm
                  metric = "ROC"       # performance measure
)
myModel2

# train a feed-forward neural net on train set 
myModel3 <- train(y ~ .,               # model specification
                  data = train,        # train set used to build model
                  method = "nnet",     # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  tuneLength = 1:5,   # how many tuning parameter combos to try
                  maxit = 100,         # max # of iterations
                  metric = "ROC"       # performance measure
)
myModel3

# train a feed-forward neural net on the down-sampled train set using a customer
# tuning parameter grid
myGrid <-  expand.grid(size = c(10,15,20)     # number of units in the hidden layer.
                       , decay = c(.09,0.12))  #parameter for weight decay. Default 0.
myModel4 <- train(y ~ .,              # model specification
                  data = dnTrain,       # train set used to build model
                  method = "nnet",    # type of model you want to build
                  trControl = ctrl,   # how you want to learn
                  tuneGrid = myGrid,  # tuning parameter combos to try
                  maxit = 100,        # max # of iterations
                  metric = "ROC"      # performance measure
)
myModel4

################################################################################
# there are so many different types of models you can try, go here to see them all
# http://topepo.github.io/caret/available-models.html
################################################################################
# Capture the train and test estimated probabilities and predicted classes
# model 1 
logit1_trp <- predict(myModel1, newdata=train, type='prob')[,1]
logit1_trc <- predict(myModel1, newdata=train)
logit1_tep <- predict(myModel1, newdata=test, type='prob')[,1]
logit1_tec <- predict(myModel1, newdata=test)
# model 2 
logit2_trp <- predict(myModel2, newdata=dnTrain, type='prob')[,1]
logit2_trc <- predict(myModel2, newdata=dnTrain)
logit2_tep <- predict(myModel2, newdata=test, type='prob')[,1]
logit2_tec <- predict(myModel2, newdata=test)
# model 3
nn1_trp <- predict(myModel3, newdata=train, type='prob')[,1]
nn1_trc <- predict(myModel3, newdata=train)
nn1_tep <- predict(myModel3, newdata=test, type='prob')[,1]
nn1_tec <- predict(myModel3, newdata=test)
# model 4 
nn2_trp <- predict(myModel4, newdata=dnTrain, type='prob')[,1]
nn2_trc <- predict(myModel4, newdata=dnTrain)
nn2_tep <- predict(myModel4, newdata=test, type='prob')[,1]
nn2_tec <- predict(myModel4, newdata=test)

################################################################################
# Now use those predictions to assess performance on the train set and testing
# set. Be on the lookout for overfitting
# model 1 - logit
(cm <- confusionMatrix(data=logit1_trc, train$y))
(testCM <- confusionMatrix(data=logit1_tec, test$y))
# model 2 - logit with down-sampled data
(cm2 <- confusionMatrix(data=logit2_trc, dnTrain$y))
(testCM2 <- confusionMatrix(data=logit2_tec, test$y))
# model 3 - nnet
(cm3 <- confusionMatrix(data=nn1_trc, train$y))
(testCM3 <- confusionMatrix(data=nn1_tec, test$y))
# model 4 - nnet with down-sampled data
(cm4 <- confusionMatrix(data=nn2_trc, dnTrain$y))
(testCM4 <- confusionMatrix(data=nn2_tec, test$y))

