
# This script has three parts: 

# 1. Training a functional ML model (you can run the scrape again to get the data, or import the scraped and 
# cleaned data ready to go from one of the following:
# all_cats <- read.csv("all_cats.csv", header=TRUE, sep=",")
# all_cats <- readRDS("all_cats.csv")


######################################################################
##### Train model for use in classification of Bunnings products #####
######################################################################

# Notes: 
# The first part of this script (the training of a ML model) takes a while (some lines take 10+ minutes), but it doesn't require any exiting of RStudio (16 GB RAM is enough)
# The whole thing works when read.csv doesn't have stringsAsFactors = FALSE, hopefully works fine with set to TRUE
# Use cluster whereever possible as it improves times (i.e. on tdm's and trains, no need on predict)

library(caret)
library(tm)
library(SnowballC)
library(arm)
library(doParallel)
library(dplyr)
Sys.setenv(TZ="Australia/Sydney")
tryCatch(setwd("C:/Users/Administrator"), error=function(e) {})

all_cats <- read.csv("C:/Users/Steve Condylios/Documents/Amazon/Training ML model for bunnings product two factor classification/all_cats_clean_for_ML_20180105.csv", header=TRUE)

# Inspect
rel <- all_cats[all_cats$one_if_rel == 1,] 
irrel <- all_cats[all_cats$one_if_rel == 0,] 

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


# Idea from web
names(train) <- make.names(names(train)); gc()

# Feature Engineering

# Remove "chrome" as it results in a LOT of false positives
train$chrome <- NULL


# Train model:

 cl <- makeCluster(detectCores()-1)
 registerDoParallel(cl)
fit <- train(one_if_rel ~ ., data = train, method = 'LogitBoost'
             ,model = FALSE # Added to try to reduce the function holding raw data and models. See: https://stackoverflow.com/questions/6543999/why-is-caret-train-taking-up-so-much-memory
)
 stopCluster(cl)


# Check accuracy on training. # Note: when predicting for test or other new datasets, only keep the columns that are contained in the fit (i.e. fit[[11]][3][[1]])
# relevant_cols <- fit[[11]][3][[1]]
a <- predict(fit, newdata = train)

# Check accuracy - 87% accurate: 
b <- cbind(train, a)
b[b$one_if_rel == b$a, ] %>% nrow(.) / nrow(b)

# SEE BOTTOM OF NEXT SECTION (TESTING SECTION) FOR SAVING RELEVANT FILES

######################################################
##### Test model accuracy on new data (test set) #####
######################################################

# In this case we'll test on all 63k bunnings products
all_cats <- read.csv("C:/Users/Steve Condylios/Documents/scraping_bunnings/all_cats_clean_for_ML_20180105.csv", header=TRUE, stringsAsFactors = FALSE)

data_63312 <- all_cats$product
corpus_63312 <- VCorpus(VectorSource(data_63312))

# for test or subsequent sets use: 
# note: had many unpleasant errors here along the lines of: Error in simple_triplet_matrix(i, j, v, nrow = length(terms), ncol = length(corpus), : 'i, j' invalid
# to fix, I install.packages("") all the relevant packages
tdm_subsequent_data <- DocumentTermMatrix(corpus_63312, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE)) 
train <- as.matrix(tdm_subsequent_data)
train <- as.data.frame(train)
names(train) <- make.names(names(train)); gc()

a <- predict(fit, newdata = train)

# Check accuracy - 95% accurate (small improvement to 95.33% when removing "chrome"):

all_cats <- cbind(all_cats, a)
all_cats[all_cats$one_if_rel == all_cats$a, ] %>% nrow(.) / nrow(all_cats)

# how many new possibly relevant products? 800. After feature-engineering out "chrome" ML identifies 426 relevant products outside of obvious categories
# This is far fewer than the ~800 it initially identified, but it is now far more accurate
all_cats[all_cats$one_if_rel == 0 & all_cats$a == 1,] %>% nrow()



# Prepare and save files

all_cats <- all_cats[, c(1,2,7,8)]
all_cats$a <- as.numeric(as.character(all_cats$a))
colnames(all_cats)[4] <- "ML_one_if_rel"
# all_cats <- all_cats[!duplicated(all_cats), ]


# setwd("C:/Users/Steve Condylios/Documents/Amazon/Training ML model for bunnings product two factor classification//Feature Engineering")
saveRDS(all_cats, "bunnings_pre_labelled.rds")
saveRDS(tdm, "tdm.rds") # original tdm necessary to use in Terms(tdm) in creating any new tdm - may be a more efficient way of storing this info (for this ML model it's around half a gig)
saveRDS(fit, "bunnings_text_classification_fit.rds")


###############################################################
##### For reference: Code used in automated scrape script #####
###############################################################



###############################
##### Bunnings Web Scrape #####
###############################



bunnings_time <- Sys.time()
library(rvest)

bun_main = read_html("https://www.bunnings.com.au/")

#dropdown_categories <- 

top_cats <- html_nodes(bun_main, "div.grid_3")[2] %>% html_nodes("li a") %>% html_attr("href") %>% .[3:11]

dropdown_categories <- c()
for (i in 1:length(top_cats)) {
  d <- read_html(paste("https://www.bunnings.com.au", top_cats[i], sep="")) %>% html_nodes("a.category-block-heading__title") %>% 
    html_attr("href")
  dropdown_categories <- c(dropdown_categories, d)
}



dropdown_categories <- as.data.frame(dropdown_categories)

f42 <- function(x) { paste("https://www.bunnings.com.au",x,sep="") }
dropdown_categories <- sapply(dropdown_categories,f42)
#dropdown_categories <- as.data.frame(dropdown_categories)

all_cats <- data.frame(product=as.character(), brand=as.character(), category=as.character(), price=as.numeric(), link=as.character(), time_of_scrape=as.character(), stringsAsFactors=FALSE)
items_complete <- 0

# 4 categories have no products to scrape (2 are informative pages; 2 are gift cards)
# dropdown_categories <- dropdown_categories[!grepl(paste(
#   c("^https://www.bunnings.com.au/our-range/kitchen$"
#     , "^https://www.bunnings.com.au/our-range/bathroom-plumbing$"
#     , "^https://www.bunnings.com.au/gift-ideas/corporate-gift-cards$" 
#     , "^https://www.bunnings.com.au/gift-ideas/gift-cards$"
#     , "^https://www.bunnings.com.au/gift-ideas/interest$"
#     , "^https://www.bunnings.com.au/gift-ideas/gifts-by-occasion$"
#     , "^https://www.bunnings.com.au/gift-ideas/gifts-by-price$"
#   )
#   , collapse = "|")
#   , dropdown_categories
# )]


for (cat in dropdown_categories) {
  
  # Go to page with all products in sub category
  splash <- read_html(cat)
  bunnings_section_url <- html_attr(html_nodes(splash,'a.view-more'), "href")
  bunnings_section_url <- paste("https://www.bunnings.com.au",bunnings_section_url,sep="")
  
  data <- data.frame(product=as.character(), brand=as.character(), category=as.character(), price=as.numeric(), link=as.character(), time_of_scrape=as.character(), stringsAsFactors=FALSE)
  x <- data.frame(product=as.character(), brand=as.character(), category=as.character(), price=as.numeric(), link=as.character(), time_of_scrape=as.character(), stringsAsFactors=FALSE)
  
  count_html <- read_html(paste(bunnings_section_url,"&page=1", sep=""))
  count <- html_text(html_nodes(count_html,'.count-block__header'))
  count_num <- as.numeric(count)
  pages_to_scrape <- ceiling(count_num/48)
  
  for (page in 1:pages_to_scrape) {
    
    bunnings <- read_html(paste(bunnings_section_url,"&page=",as.character(page), sep=""))
    
    containers <- html_nodes(bunnings, 'a.product-list__link')
    r = 1
    for (container in containers) {
      
      tryCatch(x[r,"product"] <- gsub(",","-",html_attr(html_nodes(container, 'img.photo.lazy'), "alt")), error=function(e) {x[r,"product"] <- "NA"})
      tryCatch(x[r,"brand"] <-html_attr(html_nodes(container, 'img.product-list__logo.brand'), "alt"), error=function(e) {x[r,"brand"] <- "NA"})
      x[r,"category"] <- as.character(cat)
      tryCatch(x[r,"price"] <- html_text(html_nodes(container, 'div.price-value')), error=function(e) {x[r,"price"] <- "NA"})
      x[r,"price"] <- gsub("\\$","",x[r,"price"])
      x[r,"price"] <- gsub(",","",x[r,"price"])
      x[r,"price"] <- as.numeric(x[r,"price"])
      tryCatch(x[r,"link"] <- paste("https://www.bunnings.com.au", html_attr(container, "href"), sep = ""), error = function(e) { x[r, "link"] <- "NA" })
      x[r,"time_of_scrape"] <- as.POSIXct(as.numeric(Sys.time()),origin="1970-01-01",tz=Sys.timezone())
      r = r + 1
      items_complete <- items_complete + 1
    }
    
    x[,"price"] <- as.numeric(x[,4])
    x[,"time_of_scrape"] <- as.POSIXct(round(as.numeric(x[,6]),0),origin="1970-01-01",tz=Sys.timezone())
    
    data <- rbind(data,x)
  }
  
  all_cats <- rbind(all_cats,data)
}


# remove duplicates
all_cats <- all_cats[!(duplicated(all_cats[,c(1,2)])),] 

# Take product category from category url
f45 <- function(cat) { substr(cat, as.numeric(as.character(gregexpr(pattern ='/',cat)[[1]]))[4]+1, nchar(cat)) }
all_cats$category <- mapply(f45,all_cats$category)

# Define function to capitalize 
simpleCap <- function(x) {
  s <- strsplit(x, " ")[[1]]
  paste(toupper(substring(s, 1,1)), substring(s, 2),
        sep="", collapse=" ")
}



# Tidy category columnn
f_beautify <- function(category_to_beautify) { simpleCap(gsub("/"," - ",gsub("-"," ",unique(category_to_beautify)))) }
all_cats$category <- mapply(f_beautify, all_cats$category)

# Remove commas and quotes
all_cats[] <- lapply(all_cats, gsub, pattern = ",", replacement = ";")
all_cats[] <- lapply(all_cats, gsub, pattern = '"', replacement = "INCH")
all_cats[] <- lapply(all_cats, gsub, pattern = "'", replacement = "FT")

Sys.time() - bunnings_time



# Label products relevant if in a relevant category
f_label <- function(category) { ifelse(category == "Garden - Watering Accessories" || category == "Outdoor Living - Swimming Pools Spa"
                                       || category == "Kitchen - Kitchen Taps Sinks" || category == "Kitchen - Splashbacks"
                                       || category == "Bathroom Plumbing - Bathroom" || category == "Bathroom Plumbing - Plumbing",
                                       1,0) }

all_cats$one_if_rel <- mapply(f_label, all_cats$category)

all_cats <- all_cats[!duplicated(all_cats[,c(1,2)]), ]


# Find relevant products from outside obvious categories
if (bunnings_ML == "On") {
  
  bunnings_pre_labelled <- readRDS("bunnings_pre_labelled.rds")
  bunnings_pre_labelled <- bunnings_pre_labelled[!duplicated(bunnings_pre_labelled[,c(1,2)]), ]
  
  all_cats <- left_join(all_cats, bunnings_pre_labelled, by = "product", suffix = c("", ".y"))
  all_cats <- all_cats[,c(1:7,10)]
  
  # How many extra products found? 773
  all_cats[all_cats$one_if_rel == 0 & all_cats$ML_one_if_rel == 1 & !is.na(all_cats$ML_one_if_rel), ] %>% nrow()
  all_cats$one_if_rel <- ifelse(all_cats$one_if_rel == 1 | all_cats$ML_one_if_rel == 1, 1, 0)
  
  
  # predict for any products that were not in the pre labelled data set
  all_cats_ML_new <- all_cats[is.na(all_cats$ML_one_if_rel), ] 
  
  data <- all_cats_ML_new$product
  corpus <- VCorpus(VectorSource(data))
  
  fit <- readRDS("bunnings_text_classification_fit.rds")
  tdm <- readRDS("tdm.rds")
  
  # note: had many unpleasant errors here along the lines of: Error in simple_triplet_matrix(i, j, v, nrow = length(terms), ncol = length(corpus), : 'i, j' invalid
  # to fix, install.packages("") all the relevant packages
  
  if (detectCores()-1 > 1) {
    cl <- makeCluster(detectCores()-1)
    registerDoParallel(cl)
    tdm_new_bunnings_data <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE)) 
    stopCluster(cl)
  }
  
  if (!(detectCores()-1 > 1)) {
    tdm_new_bunnings_data <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE)) 
  }
  
  train <- as.matrix(tdm_new_bunnings_data)
  train <- as.data.frame(train)
  names(train) <- make.names(names(train)); gc()
  
  predictions <- predict(fit, newdata = train)
  
  # Check accuracy - 95% accurate: 
  all_cats_ML_new <- cbind(all_cats_ML_new, predictions)
  all_cats_ML_new$predictions <- as.numeric(as.character(all_cats_ML_new$predictions))
  
  all_cats <- left_join(all_cats, all_cats_ML_new, by = "product", suffix = c("", ".y"))
  all_cats <- all_cats[,c(1:8, 16)]
  all_cats$one_if_rel <- ifelse((all_cats$one_if_rel == 1 & !is.na(all_cats$one_if_rel))| (all_cats$predictions == 1 & !is.na(all_cats$predictions)), 1, 0)
  all_cats$ML_one_if_rel <- ifelse((all_cats$ML_one_if_rel == 1 & !is.na(all_cats$ML_one_if_rel)) | (all_cats$predictions == 1 & !is.na(all_cats$predictions)), 1, 0)
  all_cats <- all_cats[, c(1:8)]
}


if (bunnings_ML != "On") {
  all_cats$ML_one_if_rel <- rep(0, nrow(all_cats))
}


write.csv(all_cats, paste("bunnings_", date_for_file_name, ".csv", sep=""), row.names = FALSE)







