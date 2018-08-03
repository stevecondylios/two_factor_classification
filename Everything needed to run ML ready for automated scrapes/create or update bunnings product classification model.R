

######################################################################
##### Train model for use in classification of Bunnings products #####
######################################################################

#----- How to run this script -----#
# Script belongs in: "two_factor_classification/Everything needed to run ML ready for automated scrapes"
# Place most recent bunnings scraped data in the directory, rename it bunnings_training_data.csv
# Run the script! 
# The 'Train the model' section will run in parallel (so it might be wise to limit usage of other things on your PC/Mac while it runs) - that part takes ~20 mins, the rest only a few mins
# The script will produce 3 output files, which can be cut/paste directly into the competitor-pricing directory (they'll be used by the scrape)

#----- Tips -----#
# Use cluster whereever possible as it improves times (i.e. on tdm's and trains, no need on predict)

library(caret)
library(tm)
library(SnowballC)
library(arm)
library(doParallel)
library(dplyr)
Sys.setenv(TZ="Australia/Sydney")

all_cats <- read.csv("bunnings_training_data.csv", header=TRUE, stringsAsFactors = FALSE)
all_cats$product <- enc2utf8(all_cats$product)

rel <- all_cats[all_cats$one_if_rel == 1, ] 
irrel <- all_cats[all_cats$one_if_rel == 0, ] 

# get a sample of 5000 from each of relevant and irrelevant 
set.seed(1000)
rel_5k <- rel[sample(nrow(rel), 5000), ]
set.seed(1000)
irrel_5k <- irrel[sample(nrow(irrel), 5000), ]

all_cats.training <- rbind(rel_5k, irrel_5k)

# Training data.
data <- all_cats.training$product
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)

# These objects can go now
# rm(data, corpus, tdm); gc()
train <- cbind(train, all_cats.training$one_if_rel)
colnames(train)[ncol(train)] <- "one_if_rel"
train <- as.data.frame(train)
train$one_if_rel <- as.factor(train$one_if_rel)


# Idea from web - prevents errors from unusual terms attempting to make their way into column names - may be alleviated by enc2utf8 but have not tested
names(train) <- make.names(names(train)); gc()

# Feature Engineering

# Remove "chrome" as it results in a LOT of false positives
train$chrome <- NULL


#----- Train the model -----#

 cl <- makeCluster(detectCores()-1)
 registerDoParallel(cl)
fit <- train(one_if_rel ~ ., data = train, method = 'LogitBoost'
             ,model = FALSE # Added to try to reduce the function holding raw data and models. See: https://stackoverflow.com/questions/6543999/why-is-caret-train-taking-up-so-much-memory
)
 stopCluster(cl)
# Note: this 'fit' object is just what we're after - it's a train model that's ready to use on new data. First, let's test on the training data:

# Check accuracy on training data. # Note: when predicting for test or other new datasets, only keep the columns that are contained in the fit (i.e. fit[[11]][3][[1]])
# relevant_cols <- fit[[11]][3][[1]]
predictions_on_training_set <- predict(fit, newdata = train)

# Check accuracy - 87% accurate: 
comparison <- cbind(train, predictions_on_training_set)
comparison[comparison$one_if_rel == comparison$predictions_on_training_set, ] %>% nrow(.) / nrow(comparison)



##############################################################
##### Test model accuracy on new data and saving outputs #####
##############################################################

# Note: You can use the existing model on new data if desired, simply read in a different day's bunnings scrape below
all_cats_new_data <- read.csv("bunnings_training_data.csv", header=TRUE, stringsAsFactors = FALSE)

new_data <- all_cats_new_data$product %>% enc2utf8()

corpus_new_data <- VCorpus(VectorSource(new_data))

# for test or subsequent sets use: 
# note: had many unpleasant errors here along the lines of: Error in simple_triplet_matrix(i, j, v, nrow = length(terms), ncol = length(corpus), : 'i, j' invalid
# to fix, I install.packages("") all the relevant packages
tdm_new_data <- DocumentTermMatrix(corpus_new_data, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE)) 
tdm_new_data_as_matrix <- as.matrix(tdm_new_data)
tdm_new_data_as_df <- as.data.frame(tdm_new_data_as_matrix)
names(tdm_new_data_as_df) <- make.names(names(tdm_new_data_as_df)); gc()

ML_one_if_rel <- predict(fit, newdata = tdm_new_data_as_df)

# Check accuracy - 95% accurate (small improvement to 95.33% when removing "chrome"):

all_cats_new_data <- cbind(all_cats_new_data, ML_one_if_rel)
all_cats_new_data[all_cats_new_data$one_if_rel == all_cats_new_data$ML_one_if_rel, ] %>% nrow(.) / nrow(all_cats_new_data)

# how many new possibly relevant products? 800. After feature-engineering out "chrome" ML identifies 426 relevant products outside of obvious categories
# This is far fewer than the ~800 it initially identified, but it is now far more accurate
all_cats_new_data[all_cats_new_data$one_if_rel == 0 & all_cats_new_data$ML_one_if_rel == 1,] %>% nrow()


#----- Prepare and save files -----#
all_cats_new_data <- all_cats_new_data[, c("product","brand","in_number", "one_if_rel","ML_one_if_rel")]
all_cats_new_data$ML_one_if_rel <- as.numeric(as.character(all_cats_new_data$ML_one_if_rel))
# all_cats_new_data <- all_cats_new_data[!duplicated(all_cats_new_data), ]

# Save files
saveRDS(all_cats_new_data, "bunnings_pre_labelled.rds")
saveRDS(tdm, "tdm.rds") # original tdm necessary to use in Terms(tdm) in creating any new tdm - may be a more efficient way of storing this info (for this ML model it's around half a gig)
saveRDS(fit, "bunnings_text_classification_fit.rds")



