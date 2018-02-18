

library(caret)
library(tm)
library(SnowballC)
library(arm)
library(doParallel)
library(dplyr)

setwd("C:/Users/Steve Condylios/Documents/Amazon/Training ML model for bunnings product two factor classification/Feature Engineering")

fit <- readRDS("bunnings_text_classification_fit.rds")
tdm <- readRDS("tdm.rds")

# Taking in bunnings_date.csv directly from scrape
all_cats_fresh <- read.csv("C:/Users/Steve Condylios/Documents/Amazon/Training ML model for bunnings product two factor classification/Feature Engineering/Fresh labels for matching app/bunnings_20180218.csv", header=TRUE, stringsAsFactors = FALSE)
all_cats_fresh$ML_one_if_rel <- NULL



f_label <- function(category) { ifelse(category == "Garden - Watering Accessories" || category == "Outdoor Living - Swimming Pools Spa"
                                       || category == "Kitchen - Kitchen Taps Sinks" || category == "Kitchen - Splashbacks"
                                       || category == "Bathroom Plumbing - Bathroom" || category == "Bathroom Plumbing - Plumbing",
                                       1,0) }

all_cats_fresh$one_if_rel <- mapply(f_label, all_cats_fresh$category)

all_cats_fresh <- all_cats_fresh[!duplicated(all_cats_fresh[,c(1,2)]), ]


data_fresh <- all_cats_fresh$product
corpus_fresh <- VCorpus(VectorSource(data_fresh))

# for test or subsequent sets use: 
# note: had many unpleasant errors here along the lines of: Error in simple_triplet_matrix(i, j, v, nrow = length(terms), ncol = length(corpus), : 'i, j' invalid
# to fix, I install.packages("") all the relevant packages
tdm_fresh <- DocumentTermMatrix(corpus_fresh, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE)) 
train <- as.matrix(tdm_fresh)
train <- as.data.frame(train)
names(train) <- make.names(names(train)); gc()

ML_one_if_rel <- predict(fit, newdata = train)

# Check accuracy - 95% accurate (small improvement to 95.33% when removing "chrome"):

all_cats_fresh <- cbind(all_cats_fresh, ML_one_if_rel)
all_cats_fresh[all_cats_fresh$one_if_rel == all_cats_fresh$ML_one_if_rel, ] %>% nrow(.) / nrow(all_cats_fresh)

# how many new possibly relevant products? 800. After feature-engineering out "chrome" ML identifies 389 relevant products outside of obvious categories
# This is far fewer than the ~800 it initially identified, but it is now far more accurate
all_cats_fresh[all_cats_fresh$one_if_rel == 0 & all_cats_fresh$ML_one_if_rel == 1,] %>% nrow()


all_cats_fresh[all_cats_fresh$one_if_rel ==1, ] %>% nrow() #10841
all_cats_fresh$one_if_rel <- ifelse((all_cats_fresh$one_if_rel == 1 & !is.na(all_cats_fresh$one_if_rel)) | (all_cats_fresh$ML_one_if_rel == 1 & !is.na(all_cats_fresh$ML_one_if_rel)), 1, 0)
# 10841 identified as relevant by category, 389 extras from ML, 11230 total

setwd("Fresh labels for matching app")
write.csv(all_cats_fresh, "all_cats_fresh.csv", row.names = FALSE)

