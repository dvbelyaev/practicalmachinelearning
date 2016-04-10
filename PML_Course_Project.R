library(caret)

# 1.1. Loading datasets

raw_train <- read.csv("./PML_CP_Data/pml-training.csv", row.names = 1)
raw_test <- read.csv("./PML_CP_Data/pml-testing.csv", row.names = 1)

table(raw_train$classe)

# 1.2. Select sensors' variables (belt, forearm, arm, and dumbbell).

sensors_train <- grep("_(belt|forearm|arm|dumbbell)|classe", names(raw_train))
sensors_test <- grep("_(belt|forearm|arm|dumbbell)", names(raw_test))
names(raw_train)[setdiff(sensors_train, sensors_test)]
raw_train <- raw_train[, sensors_train]
raw_test <- raw_test[, sensors_test]

# 1.3. Cleaning data

numeric_cols <- sapply(raw_train, is.numeric)
tidy_test <- raw_test[, numeric_cols]
numeric_cols["classe"] = TRUE
tidy_train <- raw_train[, numeric_cols]

na_cols <- colSums(is.na(tidy_train)) == 0
tidy_train <- tidy_train[, na_cols]
na_cols <- na_cols[names(na_cols) != "classe"]
tidy_test <- tidy_test[, na_cols]

tidy_train$classe <- as.factor(tidy_train$classe)

# 1.4. Training and test 

in_train <- createDataPartition(y = tidy_train$classe, p = .05, list = FALSE)
training <- tidy_train[in_train, ]
testing <- tidy_train[-in_train, ]

# 2.1. Gradient Boosting

set.seed(314159)

gbmGrid <- expand.grid(
        interaction.depth = c(1, 5, 10, 15),
        n.trees = (1:20) * 5,
        shrinkage = 0.1,
        n.minobsinnode = 20
)

fitControl <- trainControl(## 5-fold CV
        method = "cv",
        number = 5,
        verboseIter = TRUE)

fit_gbm <- train(classe ~ ., data = training,
        method = "gbm",
        trControl = fitControl,
        tuneGrid = gbmGrid,
        verbose = TRUE
)

fit_gbm
ggplot(fit_gbm)

test_predict_gbm <- predict(fit_gbm, testing)
confusionMatrix(test_predict_gbm, testing$classe)

# 2.2. Support Vector Machine

set.seed(314159)

svmGrid <- expand.grid(
        C = c(0.25, 0.5, 1, 2, 4, 8, 16, 32),
        sigma = c(0.001, 0.01, 0.1)
)

fitControl <- trainControl(
        method = "cv",
        number = 5,
        verboseIter = TRUE)

fit_svm <- train(classe ~ ., data = training,
                method = "svmRadial",
                trControl = fitControl,
                tuneGrid = svmGrid,
                preProc = c("center", "scale"),
                tuneLength = 8,
                verbose = TRUE)

fit_svm
ggplot(fit_svm)

test_predict_svm <- predict(fit_svm, testing)
confusionMatrix(test_predict_svm, testing$classe)

# 2.3. Random Forest

set.seed(314159)

rfGrid <- expand.grid(mtry = c(2, 4, 8, 16, 32, dim(training)[2]))

fitControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

fit_rf <- train(
        classe ~ ., data = training,
        method = "rf",
        trControl = fitControl,
        tuneGrid = rfGrid
)

fit_rf$finalModel
varImp(fit_rf)
ggplot(fit_rf)

train_predict_rf <- predict(fit_rf, tidy_train)
confusionMatrix(train_predict_rf, tidy_train$classe)


