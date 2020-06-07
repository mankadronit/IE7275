# Case Study
library(tidyverse)
library(psych)
library(caret)
library(FNN)
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
print(pc)
pc <- cbind(as.data.frame(pc$scores), df.norm$CLASS_LABEL) %>%
      rename(CLASS_LABEL = "df.norm$CLASS_LABEL")

##############################################################
## Data Mining Techniques
# Splitting data into training and validation sets
# Generate the training data indices

set.seed(2)
indices <- sample(seq_len(nrow(pc)), size = floor(0.6 * nrow(pc)))
# Get training and validation data
train_data <- pc[indices, ]
validation_data <- pc[-indices, ]

#######################
## Implementing KNN
accuracy.df <- data.frame("k" = 3:20, "accuracy" = rep(0, 18))

# Loop for K = 1 to 20
for (i in 3:20) {
  knn.pred <- knn(train = train_data[, 1:13], test = validation_data[, 1:13], 
                  cl = train_data$CLASS_LABEL, k = i)
  accuracy.df[i - 2, 2] <- confusionMatrix(knn.pred, validation_data$CLASS_LABEL)$overall[1]
  
}

# Get the K-value with the highest accuracy
best_k <- filter(accuracy.df, accuracy == max(accuracy.df$accuracy))$k
cat("The model with the best K is:", best_k, "\n")
cat("The model with K =", best_k, 
    "has accuracy:", paste0(100*filter(accuracy.df, k == best_k)$accuracy, "%"))

rm(list = c("accuracy.df", "df", "pc", "best_k", "i", "indices", "knn.pred"))

#######################
## Implementing Logistic Regression


##############################################################