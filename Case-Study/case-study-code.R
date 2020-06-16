# Case Study
library(tidyverse)
library(psych)
library(caret)
library(FNN)
library(ISLR)
library(tree)
library(randomForest)
library(neuralnet)
library(ROCR)
library(e1071)
library(gains)
library(ggplot2)

# Load phishing_websites.csv
df <- data.frame(read.csv("./data/phishing_websites.csv"))
# Remove "HttpsInHostname" column because it contains a few NAs
df <- df[, !colnames(df) %in% c("HttpsInHostname")]
df$CLASS_LABEL <- as.factor(df$CLASS_LABEL)
##############################################################
## Data Visualization

# Let's look at NumDots Histogram
ggplot(df, aes(NumDots)) +
  geom_histogram(binwidth = 1, color = "black", fill = "steelblue") +
  ggtitle("NumDots Histogram")

# Let's look at the UrlLength Histogram
ggplot(df, aes(UrlLength)) +
  geom_histogram(binwidth = 25, color = "black", fill = "steelblue") +
  ggtitle("UrlLength Histogram")

# Let's look at whether having an IP address in the Url gives us any information as
# to whether the website is a phishing website or not
ggplot(df, aes(as.factor(IpAddress), fill = CLASS_LABEL)) +
  geom_histogram(stat = "count") +
  ggtitle("IpAddress by CLASS_LABEL Barplot")



## Looks like all sites having an IP Address in the Url are phishing websites.

##############################################################
## Data Pre-processing

# Define the normalize function
normalize <- function(x) {
  return((x - min(x))) / (max(x) - min(x))
}

# Normalize the data frame
df.norm <- as.data.frame(cbind(as.data.frame(lapply(df[1:47], normalize)), 
                               df$CLASS_LABEL)) %>%
  rename(CLASS_LABEL = "df$CLASS_LABEL")


##############################################################
## Data Reduction and Transformation


# Performing PCA on the data
# Perform Scree Plot and Parallel Analysis
fa.parallel(df.norm[, 1:47], fa = "pc", n.iter = 100, show.legend = FALSE)

# Perform PCA with 13 components
pc <- principal(df.norm[, 1:47], nfactors = 13, rotate = "none", scores = TRUE)
pc <- cbind(as.data.frame(pc$scores), df.norm$CLASS_LABEL) %>%
  rename(CLASS_LABEL = "df.norm$CLASS_LABEL")

##############################################################
## Data Mining Techniques
# Splitting data into training and validation sets
# Generate the training data indices

set.seed(20)
indices <- sample(seq_len(nrow(pc)), size = floor(0.6 * nrow(pc)))
# Get training and validation data
train_data <- pc[indices, ]
validation_data <- pc[-indices, ]

levels(train_data$CLASS_LABEL) <- 
  make.names(levels(factor(train_data$CLASS_LABEL)))
levels(validation_data$CLASS_LABEL) <- 
  make.names(levels(factor(validation_data$CLASS_LABEL)))

# Also keep a set of train and validation sets without PCA
df.norm.train <- as.data.frame(lapply(df.norm[indices, ], as.numeric))
df.norm.validation <- as.data.frame(lapply(df.norm[-indices, ], as.numeric))

df.norm.train <- df.norm[indices, ]
df.norm.validation <- df.norm[-indices, ]
df.norm.train$CLASS_LABEL <- as.factor(df.norm.train$CLASS_LABEL)
df.norm.validation$CLASS_LABEL <- as.factor(df.norm.validation$CLASS_LABEL)

levels(df.norm.train$CLASS_LABEL) <- 
  make.names(levels(factor(df.norm.train$CLASS_LABEL)))
levels(df.norm.validation$CLASS_LABEL) <- 
  make.names(levels(factor(df.norm.validation$CLASS_LABEL)))



# Creating a performance list for each algorithm
performance_list <- data.frame("Model" = character(), 
                               "AUC" = numeric(), 
                               "Accuracy" = numeric())

# Helper Function to plot ROC Curve and Calculate Accuracy

evaluate_performance <- function(pred, labels, model_name) {
  # Accuracy
  pred.class <- ifelse(slot(pred.val, "predictions")[[1]] > 0.5, "X1", "X0")
  levels(pred.class) <- make.names(levels(factor(pred.class)))
  
  acc <- confusionMatrix(table(pred.class, labels))$overall[[1]] * 100
  
  # ROC Plot
  roc <- performance(pred.val, "tpr", "fpr")
  plot(roc, col = "red", lwd = 2, main = paste0(model_name, " ROC Curve"))
  abline(a = 0, b = 1)
  
  auc <- performance(pred.val, measure = "auc")
  
  temp <- data.frame("Model" = model_name, 
                     "AUC" = auc@y.values[[1]], 
                     "Accuracy" = acc)
  performance_list <<- rbind(performance_list, temp)
  print("Updated Performance List")
  
  rm(list = c("auc", "acc", "roc", "pred.class", "temp"))
}

#######################
## Implementing KNN

# Setting up train controls
repeats = 3
numbers = 10
tunel = 10

x <- trainControl(method = "repeatedcv",
                 number = numbers,
                 repeats = repeats,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)


knn.model <- train(CLASS_LABEL ~ . , data = train_data, method = "knn",
                  trControl = x,
                  metric = "ROC",
                  tuneLength = tunel)

# Look at the KNN Model
knn.model
plot(knn.model)

# get predictions for validation data
knn.pred <- predict(knn.model, validation_data, type = "prob")
pred.val <- prediction(knn.pred[, 2], validation_data$CLASS_LABEL) 


evaluate_performance(pred.val, validation_data$CLASS_LABEL, "KNN")
rm(list = c("repeats", "numbers", "tunel", "knn.model", "x", "knn.pred", 
            "pred.val"))

#######################
## Implementing Logistic Regression
# On PCA Dataset
glm.fit.pc <- glm(CLASS_LABEL ~ ., data = train_data, family = binomial)

glm.probs.pc <- predict(glm.fit.pc, newdata = validation_data, type = "response")
pred.val <- prediction(glm.probs.pc, validation_data$CLASS_LABEL) 

evaluate_performance(pred.val, validation_data$CLASS_LABEL, 
                     "Logistic Regression (PCA)")

# On Original Dataset
glm.fit <- glm(CLASS_LABEL ~ ., data = df.norm.train, family = binomial)

glm.probs <- predict(glm.fit, newdata = df.norm[-indices, ], type = "response")
pred.val <- prediction(glm.probs, validation_data$CLASS_LABEL) 

evaluate_performance(pred.val, validation_data$CLASS_LABEL, 
                     "Logistic Regression")

rm(list = c("glm.fit", "glm.probs",
            "glm.fit.pc", "glm.probs.pc", 
            "pred.val"))

#######################
## Implementing Naive Bayes
nb <- naiveBayes(CLASS_LABEL ~ ., data = train_data)

nb.pred <- predict(nb, newdata = validation_data, type = "raw")
pred.val <- prediction(nb.pred[, 2], validation_data$CLASS_LABEL) 

evaluate_performance(pred.val, validation_data$CLASS_LABEL, "Naive Bayes (PCA)")

rm(list = c("nb", "nb.pred", "pred.val"))

#######################
## Implementing Decision Tree
# Classification tree on PCA Dataset
tree.pca <- tree(CLASS_LABEL ~ ., data = train_data)
plot(tree.pca)
text(tree.pca, pretty = 0)

tree.pca.pred <- predict(tree.pca, validation_data)
pred.val <- prediction(tree.pca.pred[, 2], validation_data$CLASS_LABEL) 


evaluate_performance(pred.val, 
                     validation_data$CLASS_LABEL, 
                     "Classification Tree (PCA)")

# Classification tree on Original Dataset
tree <- tree(CLASS_LABEL ~ ., data = df.norm[indices, ])
plot(tree)
text(tree, pretty = 0)

tree.pred <- predict(tree, df.norm[-indices, ])
pred.val <- prediction(tree.pred[, 2], validation_data$CLASS_LABEL)

evaluate_performance(pred.val, 
                     validation_data$CLASS_LABEL, 
                     "Classification Tree")

rm(list = c("tree.pca", "tree.pca.pred", "tree", 
            "tree.pred", "pred.val"))

#######################
## Implementing Random Forests

# On PCA dataset
rf.pca <- randomForest(CLASS_LABEL ~ ., data = train_data)
rf.pca.pred <- predict(rf.pca, validation_data, type = "prob")
pred.val <- prediction(rf.pca.pred[, 2], validation_data$CLASS_LABEL)

evaluate_performance(pred.val, 
                     validation_data$CLASS_LABEL, 
                     "Random Forest (PCA)")
# On original dataset
rf <- randomForest(CLASS_LABEL ~ ., data = df.norm[indices, ])
rf.pred <- predict(rf, df.norm[-indices, ], type = "prob")
pred.val <- prediction(rf.pred[, 2], validation_data$CLASS_LABEL)

evaluate_performance(pred.val, 
                     validation_data$CLASS_LABEL, 
                     "Random Forest")


rm(list = c("rf.pca", "rf.pca.pred", "rf", 
            "rf.pred", "pred.val"))

#######################
## Implementing Artificial Neural Networks
# On PCA Dataset
nn.pca <- neuralnet(CLASS_LABEL ~ ., 
                    data = train_data, 
                    hidden = 3, 
                    act.fct = "logistic", 
                    linear.output = FALSE)

plot(nn.pca)

nn.pca.pred <- neuralnet::compute(nn.pca, validation_data[, 1:13])$net.result
pred.val <- prediction(nn.pca.pred[, 2], validation_data$CLASS_LABEL)
evaluate_performance(pred.val, validation_data$CLASS_LABEL, 
                     "Artificial Neural Net (PCA)")

# On Original Dataset
repeats = 2
numbers = 2
tunel = 6

x <- trainControl(method = "repeatedcv",
                  number = numbers,
                  repeats = repeats,
                  classProbs = TRUE,
                  summaryFunction = twoClassSummary)


nn <- train(CLASS_LABEL ~ . , data = df.norm.validation, 
            method = "nnet",
            trControl = x,
            metric = "ROC",
            tuneLength = tunel)

nn.pred <- predict(nn, newdata = df.norm.validation, type = "prob")
pred.val <- prediction(nn.pred[, 2], df.norm.validation$CLASS_LABEL)
evaluate_performance(pred.val, df.norm.validation$CLASS_LABEL, 
                     "Artificial Neural Net")


rm(list = c("nn.pca", "nn.pca.pred", "nn", "pred.val", "nn.pred"))
##############################################################

write.csv(performance_list, "performance_list.csv")
