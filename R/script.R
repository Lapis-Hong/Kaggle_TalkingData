if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, caTools, tidyverse, data.table, lubridate, tictoc, DescTools, lightgbm)
set.seed(84)               
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "../input/train.csv"
test_path  <- "../input/test.csv"

tr_col_names <- c("ip", "app", "device", "os", "channel", "click_time", "attributed_time", 
                  "is_attributed")
most_freq_hours_in_test_data <- c("4","5","9","10","13","14")
least_freq_hours_in_test_data <- c("6","11","15")

#*****************************************************************
#Feature engineering

train_chunker <- function(chunk, skiprows, nrows) {
    cat("Piping train: ", chunk, "\n")
    df_name <- fread(train_path, skip=skiprows, nrows=nrows, colClasses=list(numeric=1:5),
                     showProgress = FALSE, col.names = tr_col_names) %>%
      select(-c(attributed_time)) %>%
      mutate(wday = Weekday(click_time), 
         hour = hour(click_time),
         in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                          ifelse(hour %in% least_freq_hours_in_test_data, 2, 3))) %>%
      select(-c(click_time)) %>%
      add_count(ip, wday, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
      select(-c(in_test_hh)) %>%
      add_count(ip, wday, hour) %>% rename("nip_day_hh" = n) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip))
      return(df_name)
  }

#*****************************************************************
#Process chunk size of 75 million rows from training data

total_rows <- 184903890
chunk_rows <- 75000000
skip1_rows <- total_rows - chunk_rows 

tic("Total processing time for train data --->")
train <- train_chunker("Chunk of 75 million rows", skip1_rows, chunk_rows)
dim(train)
print(object.size(train), units = "auto")
toc()
invisible(gc())
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
print("free up memory by converting to lgb.Datast before reding test data")
#time based split
# tr_index <- nrow(train)
# dtrain <- train %>% head(0.95 * tr_index)
# valid  <- train %>% tail(0.05 * tr_index)

#shuffled split
sample = sample.split(train$is_attributed, SplitRatio = 0.95)
dtrain = subset(train, sample == TRUE)
valid  = subset(train, sample == FALSE)

print("Table of class unbalance")
table(dtrain$is_attributed)

rm(train)
invisible(gc())

cat("train size : ", dim(dtrain), "\n")
cat("valid size : ", dim(valid), "\n")

categorical_features = c("app", "device", "os", "channel", "hour")

dtrain = lgb.Dataset(data = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), 
                     label = dtrain$is_attributed, categorical_feature = categorical_features)
dvalid = lgb.Dataset(data = as.matrix(valid[, colnames(valid) != "is_attributed"]), 
                     label = valid$is_attributed, categorical_feature = categorical_features)

rm(valid)
invisible(gc())

cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
print("Piping test data:")

tic("Total processing time for test data --->")
test <- fread(test_path, colClasses=list(numeric=2:6), showProgress = FALSE)
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())

test <- test %>%
      mutate(wday = Weekday(click_time), 
         hour = hour(click_time),
         in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                          ifelse(hour %in% least_freq_hours_in_test_data, 2, 3))) %>%
      select(-c(click_time)) %>%
      add_count(ip, wday, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
      select(-c(in_test_hh)) %>%
      add_count(ip, wday, hour) %>% rename("nip_day_hh" = n) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip))

print(object.size(test), units = "auto")
cat("test  size : ", dim(test), "\n")
toc()

dtest <- as.matrix(test[, colnames(test)])


cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
#Modelling

print("Modelling")
params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.1, 
              num_leaves= 7,
              max_depth= 4,
              min_child_samples= 100,
              max_bin= 100,
              subsample= 0.7,
              subsample_freq= 1,
              colsample_bytree= 0.7,
              min_child_weight= 0,
              min_split_gain= 0,
              scale_pos_weight=99.7)

tic("Total time for model training --->")
model <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                   nrounds = 500, verbose= 1, early_stopping_rounds = 50, eval_freq = 25)

rm(dtrain, dvalid)
invisible(gc())
toc()
cat("Validation AUC @ best iter: ", max(unlist(model$record_evals[["validation"]][["auc"]][["eval"]])))

#*****************************************************************
#Predictions

print("Predictions")
preds <- predict(model, data = dtest, n = model$best_iter)
preds <- as.data.frame(preds)
sub$is_attributed = preds
sub$is_attributed = round(sub$is_attributed,4)
fwrite(sub, "sub_lightgbm_R_75m.csv")
head(sub,10)

print("Feature importance")
kable(lgb.importance(model, percentage = TRUE))

print("finished")
cat(paste(" 


# What's new in this kernel?
- shuffled split is used in this kernel instead of time based split to measure any performance improvements

Check out the post submission analysis section at the end of my EDA report (link below) where I have compared model performance for different test runs by:
- data processing and modelling time
- gain improvement by features
- validation auc vs LB score

link: https://www.kaggle.com/pranav84/r-lightgbm-eda-to-model-evaluation-lb-0-9683



"))