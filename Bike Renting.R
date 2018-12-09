rm(list = ls())

library(usdm)
library(rpart)
library(caret)
library(randomForest)

read.csv("./day.csv") -> bike.data
str(bike.data)

######################### PREPROCESSING #########################

bike.data <- bike.data[,!names(bike.data) %in% c('instant')]

bike.data[, "season"] <- as.factor(bike.data[, "season"])
bike.data[, "yr"] <- as.factor(bike.data[, "yr"])
bike.data[, "mnth"] <- as.factor(bike.data[, "mnth"])
bike.data[, "holiday"] <- as.factor(bike.data[, "holiday"])
bike.data[, "weekday"] <- as.factor(bike.data[, "weekday"])
bike.data[, "workingday"] <- as.factor(bike.data[, "workingday"])
bike.data[, "weathersit"] <- as.factor(bike.data[, "weathersit"])
bike.data[, "dteday"] <-
    as.Date(as.character(bike.data[, "dteday"]))

######################### MISSING VALUE ANALYSIS #########################

for (i in colnames(bike.data)) {
    print(i)
    print(sum(is.na(bike.data[i])))
}
rm(i)

numerical <- colnames(bike.data[, sapply(bike.data, is.numeric)])
categorical <- colnames(bike.data[, sapply(bike.data, is.factor)])

######################### OUTLIER ANALYSIS #########################

for (i in numerical) {
    print(i)
    val <- 
        bike.data[, i][bike.data[, i] %in% boxplot.stats(bike.data[, i])$out]
    print(val)
    
    boxplot(bike.data[, i])# plot = i)
    
    val[val < boxplot.stats(bike.data[, i])$stats[1]] -> val.below
    val[val > boxplot.stats(bike.data[, i])$stats[5]] -> val.above
    
    print(paste0("The outlier above the maximum : ", length(val.above)))
    print(paste0("The outlier below the minimum : ", length(val.below)))
    cat("\n\n")
}
rm(list = c('i', 'val.above', 'val.below', 'val'))

######################### FEATURE SELECTION #########################
X <- bike.data
y <- X$cnt
drop.list <- c('dteday')
X <- X[, !names(X) %in% drop.list]

vif(X)
drop.list <- c( 'temp', 'casual', 'registered')
X <- X[, !names(X) %in% drop.list]

vif(X)

rm(list = c('drop.list'))
######################### FEATURE SCALING #########################
numerical <-  colnames(X[, sapply(X, is.numeric)])

for (i in numerical) {
    print(i)
    X[, i] <-
        (X[, i] - mean(X[, i])) / sd(X[, i])
}
rm(i)
######################### MODEL DEVELOPMENT #########################

train.index <- sample(1:nrow(X), 0.8 *nrow(X))
train.X <- X[train.index,]
test.X <- X[-train.index,]
test.y <- test.X$cnt
test.X <- test.X[, !names(test.X) %in% c('cnt')]


MAPE <- function(y, y.new) {
    mean(abs((y - y.new)/ y))
}

linear.model <- lm(cnt~., data = train.X)
predict.op <- predict(linear.model, test.X)
MAPE(test.y, predict.op)

model.new <- rpart(cnt~., data = train.X)
predict.op <- predict(model.new, test.X)
MAPE(test.y, predict.op)

model.new <- knnreg(train.X[,1:10], train.X[,11], k = 3)
predict.op <- predict(model.new, test.X)
MAPE(test.y, predict.op)

model.new <- randomForest(cnt~., train.X, importance = TRUE)
predict.op <- predict(model.new, test.X)
MAPE(test.y, predict.op)
