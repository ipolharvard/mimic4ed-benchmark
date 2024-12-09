# Install required libraries
library(AutoScore)

# Load data
# set paths to X_train, X_test, y_train and y_test .csv files, the outcome column should be named "label"
X_train <- read.csv("X_train.csv")
y_train <- read.csv("y_train.csv")
X_test <- read.csv("X_test.csv")
y_test <- read.csv("y_test.csv")

# Combine features and outcome
train_set <- cbind(X_train, label = as.integer(as.logical(y_train$label)))
test_set <- cbind(X_test, label = as.integer(as.logical(y_test$label)))
names(train_set) <- make.unique(names(train_set))
categorical_vars <- c('chiefcom_chest_pain', 'chiefcom_abdominal_pain', 'chiefcom_headache',
                      'chiefcom_shortness_of_breath', 'chiefcom_back_pain', 'chiefcom_cough',
                      'chiefcom_nausea_vomiting', 'chiefcom_fever_chills', 'chiefcom_syncope',
                      'chiefcom_dizziness')

train_set[categorical_vars] <- lapply(train_set[categorical_vars], as.factor)
check_data(train_set)

ranking <- AutoScore_rank(train_set = train_set, method = "rf", ntree = 100)

AUC <- AutoScore_parsimony(
  train_set = train_set, validation_set = test_set,
  rank = ranking, max_score = 100, n_min = 1, n_max = 15,
  categorize = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1),
  auc_lim_min = 0.5, auc_lim_max = "adaptive"
)

num_var <- 10
final_variables <- names(ranking[1:num_var])

cut_vec <- AutoScore_weighting( 
  train_set = train_set, validation_set = train_set,
  final_variables = final_variables, max_score = 100,
  categorize = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1)
)

scoring_table <- AutoScore_fine_tuning(
  train_set = train_set, validation_set = train_set, 
  final_variables = final_variables, cut_vec = cut_vec, max_score = 100
)

pred_score <- AutoScore_testing(
  test_set = test_set, final_variables = final_variables, cut_vec = cut_vec,
  scoring_table = scoring_table, threshold = "best", with_label = TRUE
)

# save predictions
write.csv(pred_score, "preds.csv")
