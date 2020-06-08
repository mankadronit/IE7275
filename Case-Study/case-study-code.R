# Case Study
library(tidyverse)
library(psych)
library(caret)
library(FNN)
library(ISLR)
library(tree)
library(randomForest)
library(e1071)
library(ggplot2)

# Load phishing_websites.csv
df <- read.csv("./data/phishing_websites.csv")
# Remove "HttpsInHostname" column
df <- df[, !colnames(df) %in% c("HttpsInHostname")]
df$CLASS_LABEL <- as.factor(df$CLASS_LABEL)

##############################################################
## Data Visualization

# Let's look at NumDots Histogram
ggplot(df, aes(NumDots)) +
  geom_histogram(binwidth = 1, color = "black", fill = "steelblue")

# Let's look at the UrlLength Histogram
ggplot(df, aes(UrlLength)) +
  geom_histogram(binwidth = 25, color = "black", fill = "steelblue")

# Let's look at whether having an IP address in the Url gives us any information as
# to whether the website is a phishing website or not
ggplot(df, aes(as.factor(IpAddress), fill = CLASS_LABEL)) +
  geom_histogram(stat = "count")



## Looks like all sites having an IP Address in the Url are phishing websites.

##############################################################
## Data Pre-processing

# Define the normalize function
normalize <- function(x) {
  return((x - min(x))) / (max(x) - min(x))
}

# Normalize the dataframe
df.norm <- cbind(as.data.frame(lapply(df[1:47], normalize)), df$CLASS_LABEL) %>%
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



# Creating a performance list for each algorithm
performance_list <- data.frame("Model" = NA, "Accuracy" = NA)

#######################
## Implementing KNN

k <- as.integer(sqrt(dim(train_data)[[1]])/2 + 1)

knn.pred <- knn(train = train_data[, 1:13], 
                test = validation_data[, 1:13], 
                cl = train_data$CLASS_LABEL, 
                k = k)

knn.pred
cf <- confusionMatrix(knn.pred, validation_data$CLASS_LABEL)

cf
# Get the K-value with the highest accuracy
knn_accuracy <- 100 * cf$overall[[1]]

cat(
  "The model with K =", k,
  "has accuracy:", paste0(knn_accuracy, "%"), "\n"
)

# Add to performance list
performance_list[1, ] <- c("KNN (PCA)", knn_accuracy)

rm(list = c("k", "knn.pred", "knn_accuracy", "cf"))

#######################
## Implementing Logistic Regression
glm.fit.pc <- glm(CLASS_LABEL ~ ., data = train_data, family = binomial)


glm.probs.pc <- predict(glm.fit.pc, newdata = validation_data, type = "response")
glm.pred.pc <- ifelse(glm.probs.pc > 0.5, 1, 0)

cf.pc <- confusionMatrix(table(glm.pred.pc, validation_data$CLASS_LABEL))

glm.fit <- glm(CLASS_LABEL ~ ., data = df.norm[indices, ], family = binomial)


glm.probs <- predict(glm.fit, newdata = df.norm[-indices, ], type = "response")
glm.pred <- ifelse(glm.probs > 0.5, 1, 0)

cf <- confusionMatrix(table(glm.pred, df.norm[-indices, ]$CLASS_LABEL))

logistic_accuracy.pc <- 100 * cf.pc$overall[1]
logistic_accuracy <- 100 * cf$overall[1]
cat("Accuracy of Logistic Regression Model with PCA:", paste0(logistic_accuracy.pc, "%"), "\n")
cat("Accuracy of Logistic Regression Model:", paste0(logistic_accuracy, "%"), "\n")


performance_list[dim(performance_list)[[1]] + 1, ] <- c("Logistic Regression (PCA)", logistic_accuracy.pc)
performance_list[dim(performance_list)[[1]] + 1, ] <- c("Logistic Regression", logistic_accuracy)

rm(list = c("glm.fit", "glm.pred", "glm.probs", "logistic_accuracy", "cf",
            "glm.fit.pc", "glm.pred.pc", "glm.probs.pc", "logistic_accuracy.pc", "cf.pc"))

#######################
## Implementing Naive Bayes
nb <- naiveBayes(CLASS_LABEL ~ ., data = train_data)

pred.class <- predict(nb, newdata = validation_data)

cf <- confusionMatrix(table(pred.class, validation_data$CLASS_LABEL))

nb_accuracy <- 100 * cf$overall[[1]]
cat("Accuracy of Naive Bayes Model:", paste0(nb_accuracy, "%"), "\n")

performance_list[dim(performance_list)[[1]] + 1, ] <- c("Naive Bayes (PCA)", nb_accuracy)

rm(list = c("nb", "cf", "pred.class", "nb_accuracy"))

#######################
## Implementing Decision Tree
# Classification tree on PCA Dataset
tree.pca <- tree(CLASS_LABEL ~ ., data = train_data)
plot(tree.pca)
text(tree.pca, pretty = 0)

tree.pca.pred <- predict(tree.pca, validation_data, type = "class")
cf.pca <- confusionMatrix(tree.pca.pred, validation_data$CLASS_LABEL)

# Classification tree on Original Dataset
tree <- tree(CLASS_LABEL ~ ., data = df.norm[indices, ])
plot(tree)
text(tree, pretty = 0)

tree.pred <- predict(tree, df.norm[-indices, ], type = "class")
cf <- confusionMatrix(tree.pred, df.norm[-indices, ]$CLASS_LABEL)

tree_pca_accuracy <- 100 * cf.pc$overall[[1]]
tree_accuracy <- 100 * cf$overall[[1]]

cat("Accuracy of Decision Tree (PCA):", paste0(tree_pca_accuracy, "%"), "\n")
cat("Accuracy of Decision Tree:", paste0(tree_accuracy, "%"), "\n")

performance_list[dim(performance_list)[[1]] + 1, ] <- c("Decision Tree (PCA)", tree_pca_accuracy)
performance_list[dim(performance_list)[[1]] + 1, ] <- c("Decision Tree", tree_accuracy)

rm(list = c("tree.pca", "tree.pca.pred", "cf.pca", "tree", 
            "tree.pred", "cf", "tree_pca_accuracy", "tree_accuracy"))

#######################
## Implementing Random Forests

# On PCA dataset
rf.pca <- randomForest(CLASS_LABEL ~ ., data = train_data)
rf.pca.pred <- predict(rf.pca, validation_data)
cf.pca <- confusionMatrix(rf.pca.pred, validation_data$CLASS_LABEL)


rf_pca_accuracy <- 100 * cf.pca$overall[[1]]

# On original dataset
rf <- randomForest(CLASS_LABEL ~ ., data = df.norm[indices, ])
rf.pred <- predict(rf, df.norm[-indices, ])
cf <- confusionMatrix(rf.pred, df.norm[-indices, ]$CLASS_LABEL)


rf_accuracy <- 100 * cf$overall[[1]]

cat("Accuracy of Random Forests (PCA):", paste0(rf_pca_accuracy, "%"), "\n")
cat("Accuracy of Random Forests:", paste0(rf_accuracy, "%"), "\n")

performance_list[dim(performance_list)[[1]] + 1, ] <- c("Random Forests (PCA)", rf_pca_accuracy)
performance_list[dim(performance_list)[[1]] + 1, ] <- c("Random Forests", rf_accuracy)

rm(list = c("rf.pca", "rf.pca.pred", "cf.pca", "rf", 
            "rf.pred", "cf", "rf_accuracy", "rf_pca_accuracy"))
##############################################################
