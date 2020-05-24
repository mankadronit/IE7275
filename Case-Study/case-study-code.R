# Case Study
library(tidyverse)
library(psych)

df <- read.csv("./data/phishing_websites.csv")

df <- df[, !colnames(df) %in% c("HttpsInHostname")]

fa.parallel(scale(df[, 1:47]), fa = "pc", n.iter = 100, show.legend = FALSE)

pc <- principal(scale(df[, 1:47]), nfactors = 13, rotate = "none", scores = TRUE)

biplot.psych(pc, choose = c(1, 2), labels = df$CLASS_LABEL)
