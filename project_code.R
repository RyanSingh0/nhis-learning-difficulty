## MET CS699 – Final Project
## Team: Aryan Meena + Aman Nishad
## Data: project_data.csv

set.seed(699)

#Some models takes time to be learned, so please wait.
#Please install all the libraries needed.
#Libraries 

lapply(c(
  "tidyverse","caret","rsample","pROC","xgboost","naivebayes",
  "rpart","rpart.plot","randomForest","nnet","e1071","ipred","kernlab"
), library, character.only = TRUE)

#Load data 
#NHIS-like missing codes are marked as NA at read time
raw <- read.csv("project_data.csv",
                na.strings = c("", " ", "NA", "NaN", ".", "99", "999", "9999",
                               "-9","-99","-999", "7","8","9","97","98","99"))
stopifnot("Class" %in% names(raw))
raw$Class <- factor(raw$Class, levels = c("Yes","No"))  # positive class first

#Preprocessing
df <- raw

#1) Drop columns with ≥90% missing
missrate <- sapply(df, function(x) mean(is.na(x)))
drop_cols <- names(missrate)[missrate >= 0.90]
if (length(drop_cols)) df <- df[, setdiff(names(df), drop_cols), drop = FALSE]

#2) Removing near-zero-variance predictors
nzv <- nearZeroVar(df, saveMetrics = TRUE)
if (any(nzv$nzv)) df <- df[, !nzv$nzv, drop = FALSE]

#3) Type splits
is_num <- sapply(df, is.numeric)
num_cols <- names(df)[is_num & names(df) != "Class"]
cat_cols <- setdiff(names(df), c(num_cols, "Class"))


#4) Simple imputation (median for numeric; mode for factors)
for (nm in num_cols) {
  df[[nm]][is.na(df[[nm]])] <- median(df[[nm]], na.rm = TRUE)
}
mode_impute <- function(v){
  v <- as.character(v)
  if (all(is.na(v))) return(factor(v))
  m <- names(sort(table(v), decreasing = TRUE))[1]
  v[is.na(v)] <- m
  factor(v)
}
for (nm in cat_cols) df[[nm]] <- mode_impute(df[[nm]])

#5) Winsorize numeric at 1%/99%
winsor <- function(x){ q <- quantile(x, c(.01,.99), na.rm = TRUE)
x[x < q[1]] <- q[1]; x[x > q[2]] <- q[2]; x
}
for (nm in num_cols) df[[nm]] <- winsor(df[[nm]])

#Drop ID-like columns
id_regex <- "(^|_)(id|hhx|se?qn|case|psu|strata)(_|$)"
is_id_name <- grepl(id_regex, names(df), ignore.case = TRUE)

is_high_card_factor <- vapply(df, function(x) {
  if (is.factor(x) || is.character(x)) {
    u <- length(unique(x))
    u > 0.9 * nrow(df)  
  } else FALSE
}, logical(1))

drop_cols <- names(df)[is_id_name | is_high_card_factor]
if (length(drop_cols)) {
  message("Dropping ID-like columns: ", paste(drop_cols, collapse = ", "))
  df <- df[, setdiff(names(df), drop_cols), drop = FALSE]
}


#6) Correlation pruning among numerics (≥0.80) to speed/denoise (L1/L3)
if (length(num_cols) > 1) {
  X <- df[, num_cols, drop = FALSE]
  
  # pairwise correlations; avoid NA-induced failures
  cm <- suppressWarnings(cor(X, use = "pairwise.complete.obs", method = "pearson"))
  
  # replace NA
  diag(cm) <- 0
  cm[is.na(cm)] <- 0
  
  # now safely prune
  hi <- caret::findCorrelation(cm, cutoff = 0.80, names = TRUE, exact = TRUE)
  
  if (length(hi)) {
    message(sprintf("Removing %d highly correlated predictors: %s",
                    length(hi), paste(hi, collapse = ", ")))
    df <- df[, setdiff(names(df), hi), drop = FALSE]
  }
}


#Save preprocessed
write.csv(df, "preprocessed_data.csv", row.names = FALSE)

preprocessed <- read.csv("preprocessed_data.csv")
dim(preprocessed)


#PLOTS: Missingness & Class Balance
# 1A) Missing by feature
miss_df <- data.frame(
  feature = names(raw),
  missing_pct = sapply(raw, function(x) mean(is.na(x))) * 100
) %>% dplyr::arrange(dplyr::desc(missing_pct))

#bar plot of missingness
top_n <- min(40, nrow(miss_df))
ggplot(miss_df[1:top_n, ], aes(x = reorder(feature, missing_pct), y = missing_pct)) +
  geom_col() +
  coord_flip() +
  labs(title = "Percent Missing by Feature (Raw)",
       x = "Feature", y = "Missing (%)")

#1B) Class balance in RAW / TRAIN / TEST
balance_bar <- function(vec, title) {
  tb <- as.data.frame(table(vec))
  names(tb) <- c("Class","Count")
  ggplot(tb, aes(x = Class, y = Count, fill = Class)) +
    geom_col(width = 0.6) +
    labs(title = title, x = NULL, y = "Count") +
    guides(fill = "none")
}
# RAW
if ("Class" %in% names(raw)) print(balance_bar(raw$Class, "Class Balance (RAW)"))


#Split & Scale
#Stratified 80/20
set.seed(699)
split <- initial_split(df, prop = 0.80, strata = Class)
min_n <- 10  # keep levels with at least 10 rows in TRAIN

train0 <- training(split); test0 <- testing(split)
#PLOTS: Class Balance (Train/Test) & Boxplots (pre-scale)
print(balance_bar(train0$Class, "Class Balance (TRAIN)"))
print(balance_bar(test0$Class,  "Class Balance (TEST)"))

# Class-conditioned boxplots for numerics
num_cols_pre <- names(train0)[sapply(train0, is.numeric) & names(train0) != "Class"]
if (length(num_cols_pre)) {
  # showing top 8 numerics by variance to avoid huge panels
  v_ <- sapply(train0[num_cols_pre], var, na.rm = TRUE)
  top8 <- names(sort(v_, decreasing = TRUE))[1:min(8, length(v_))]
  df_long <- train0 %>%
    dplyr::select(dplyr::all_of(c("Class", top8))) %>%
    tidyr::pivot_longer(-Class, names_to = "Feature", values_to = "Value")
  ggplot(df_long, aes(x = Class, y = Value, fill = Class)) +
    geom_boxplot(outlier.alpha = 0.2) +
    facet_wrap(~ Feature, scales = "free_y") +
    labs(title = "Numeric Distributions by Class (Pre-Scaling)",
         x = NULL, y = "Value") +
    guides(fill = "none")
}



fac_cols <- setdiff(names(train0)[sapply(train0, is.factor)], "Class")

for (col in fac_cols) {
  tr_vals <- as.character(train0[[col]])
  # frequency in TRAIN
  keep_levels <- names(which(table(tr_vals) >= min_n))
  # map train
  train0[[col]] <- factor(ifelse(tr_vals %in% keep_levels, tr_vals, "Other"))
  # map test with the SAME keep set
  ts_vals <- as.character(test0[[col]])
  test0[[col]]  <- factor(ifelse(ts_vals %in% keep_levels, ts_vals, "Other"),
                          levels = levels(train0[[col]]))
}

#Drop zero-variance numerics
num_tr <- names(train0)[sapply(train0, is.numeric) & names(train0) != "Class"]

zv <- vapply(train0[, num_tr, drop = FALSE],
             function(v) sd(v, na.rm = TRUE) == 0 || length(na.omit(unique(v))) <= 1,
             logical(1))

drop_zv <- names(zv)[zv]
if (length(drop_zv)) {
  message("Dropping zero-variance numeric predictors (train-driven): ",
          paste(drop_zv, collapse = ", "))
  train0 <- train0[, setdiff(names(train0), drop_zv), drop = FALSE]
  test0  <- test0[,  setdiff(names(test0),  drop_zv), drop = FALSE]
}

#recompute numeric columns after dropping
num_tr <- names(train0)[sapply(train0, is.numeric) & names(train0) != "Class"]

## Center/scale numerics using train stats
num_tr <- names(train0)[sapply(train0, is.numeric) & names(train0) != "Class"]
pp <- if (length(num_tr)) preProcess(train0[, num_tr, drop=FALSE], method=c("center","scale")) else NULL
train_scaled <- train0; test_scaled <- test0
if (!is.null(pp)) {
  train_scaled[, num_tr] <- predict(pp, train0[, num_tr, drop=FALSE])
  test_scaled[,  num_tr] <- predict(pp, test0[,  num_tr, drop=FALSE])
}

#train/test set save
write.csv(train0, "initial_train.csv", row.names = FALSE)
write.csv(test0,  "initial_test.csv",  row.names = FALSE)

#Freeze CV folds and RNG seeds for caret
set.seed(7006)

# 3-fold stratified indices (deterministic)
folds <- caret::createFolds(train_scaled$Class, k = 3, returnTrain = TRUE)

# Helper to build a seeds list of correct length:
#   length = (#resamples) + 1
#   each resample vector length = #tuning combos
make_caret_seeds <- function(n_resamples, n_grid) {
  set.seed(7006)
  seeds <- vector("list", length = n_resamples + 1L)
  for (i in seq_len(n_resamples)) seeds[[i]] <- sample.int(1e6, n_grid)
  seeds[[n_resamples + 1L]] <- sample.int(1e6, 1L)  # final model seed
  seeds
}

make_numeric_xy <- function(df, ref_cols = NULL) {
  stopifnot("Class" %in% names(df))
  mm <- stats::model.matrix(Class ~ . - 1, data = df) |> as.data.frame()
  if (!is.null(ref_cols)) {
    miss <- setdiff(ref_cols, names(mm))
    for (m in miss) mm[[m]] <- 0
    extra <- setdiff(names(mm), ref_cols)
    if (length(extra)) mm <- mm[, setdiff(names(mm), extra), drop = FALSE]
    mm <- mm[, ref_cols, drop = FALSE]
  }
  keep <- rowSums(is.na(mm)) == 0
  list(
    x = mm[keep, , drop = FALSE],
    y = droplevels(df$Class[keep]),
    cols = colnames(mm),
    keep = keep
  )
}


# grid sizes for each model family
n_grid_glm   <- if (is.null(grid_glm)) 1L else nrow(grid_glm)
n_grid_rpart <- nrow(grid_rpart)
n_grid_nb    <- nrow(grid_nb)
n_grid_knn   <- nrow(grid_knn)
n_grid_svmR  <- nrow(grid_svmR)
n_grid_nnet  <- nrow(grid_nnet)
n_grid_rf    <- nrow(grid_rf)
n_grid_xgbB  <- nrow(grid_xgb_balanced)
n_grid_bag   <- 1L

grid_xgb_rest <- if (exists("grid_xgb") && is.data.frame(grid_xgb)) grid_xgb else grid_xgb_balanced
n_grid_xgbB   <- nrow(grid_xgb_balanced) #'cause our best model is this, so tuning it specifically
n_grid_xgbR   <- nrow(grid_xgb_rest)

#Balancing – 4 methods
#1) caret “down” (undersample) & 2) caret “up” (oversample) via trainControl
ctrl_base <- caret::trainControl(
  method = "cv", number = 3,
  index = folds,                 # fixed folds
  classProbs = TRUE,
  summaryFunction = defaultSummary,
  savePredictions = "final",
  verboseIter = FALSE,
  allowParallel = FALSE          #turn off parallel for repeatability
)

#Derived controls with sampling
ctrl_down <- ctrl_base; ctrl_down$sampling <- "down"
ctrl_up   <- ctrl_base; ctrl_up$sampling   <- "up"
ctrl_none <- ctrl_base

#3) SYN — simple synthetic oversample
make_SYN <- function(df, target = "Class", pos = "Yes", ratio = 1.0, jitter_sd = 0.02) {
  stopifnot(target %in% names(df))
  pos_ix <- which(df[[target]] == pos); neg_ix <- which(df[[target]] != pos)
  need <- max(0, as.integer(ratio*length(neg_ix) - length(pos_ix)))
  if (need == 0) return(df)
  
  num_cols <- names(df)[sapply(df, is.numeric) & names(df) != target]
  pick <- sample(pos_ix, size = need, replace = TRUE)
  syn  <- df[pick, , drop = FALSE]
  if (length(num_cols)) {
    for (nm in num_cols) {
      sdv <- sd(df[[nm]], na.rm = TRUE); if (!is.finite(sdv) || sdv == 0) next
      syn[[nm]] <- syn[[nm]] + rnorm(nrow(syn), 0, jitter_sd*sdv)
    }
  }
  out <- rbind(df, syn); rownames(out) <- NULL; out
}

#4) NM1 — NearMiss-1 style undersample (keep majority closest to minority)
make_NM1 <- function(df, target = "Class", pos = "Yes", keep_ratio = 1.0) {
  stopifnot(target %in% names(df))
  pos_df <- df[df[[target]] == pos, , drop = FALSE]
  neg_df <- df[df[[target]] != pos, , drop = FALSE]
  num_cols <- names(df)[sapply(df, is.numeric) & names(df) != target]
  if (nrow(pos_df) == 0 || nrow(neg_df) == 0) return(df)
  
  if (!length(num_cols)) {
    need <- as.integer(nrow(pos_df) * keep_ratio)
    need <- max(1, min(need, nrow(neg_df)))
    neg_keep <- neg_df[sample(nrow(neg_df), need), , drop = FALSE]
    out <- rbind(pos_df, neg_keep); rownames(out) <- NULL; return(out)
  }
  
  Xpos <- scale(pos_df[, num_cols, drop=FALSE])
  Xneg <- scale(neg_df[, num_cols, drop=FALSE],
                center = attr(Xpos,"scaled:center"),
                scale  = attr(Xpos,"scaled:scale"))
  # distance to nearest minority
  chunk <- 800
  nearest <- numeric(nrow(Xneg))
  for (i in seq(1, nrow(Xneg), by = chunk)) {
    j <- min(i + chunk - 1, nrow(Xneg))
    block <- rbind(Xneg[i:j, , drop=FALSE], Xpos)
    D <- as.matrix(dist(block))
    k <- nrow(Xneg[i:j, , drop=FALSE])
    nearest[i:j] <- apply(D[1:k, (k+1):nrow(block), drop=FALSE], 1, min)
  }
  need <- as.integer(nrow(pos_df) * keep_ratio)
  need <- max(1, min(need, nrow(neg_df)))
  neg_keep <- neg_df[order(nearest)[seq_len(need)], , drop = FALSE]
  out <- rbind(pos_df, neg_keep); rownames(out) <- NULL; out
}

#Build the two manual training sets (test stays untouched)
train_SYN <- make_SYN(train_scaled, target = "Class", pos = "Yes", ratio = 1.0)
train_NM1 <- make_NM1(train_scaled, target = "Class", pos = "Yes", keep_ratio = 1.0)
train_NM1 <- train_NM1[complete.cases(train_NM1), , drop = FALSE]
train_SYN <- train_SYN[complete.cases(train_SYN), , drop = FALSE]
train_SYN$Class <- factor(train_SYN$Class, levels = c("Yes","No"))
train_NM1$Class <- factor(train_NM1$Class, levels = c("Yes","No"))

#PLOTS: Class Counts by Balancing Scheme
count_df <- dplyr::bind_rows(
  data.frame(Scheme = "DOWN (caret)", table(caret::downSample(x = train_scaled %>% dplyr::select(-Class),
                                                              y = train_scaled$Class)$Class)),
  data.frame(Scheme = "UP (caret)",   table(caret::upSample(  x = train_scaled %>% dplyr::select(-Class),
                                                              y = train_scaled$Class)$Class)),
  data.frame(Scheme = "SYN (manual)", table(train_SYN$Class)),
  data.frame(Scheme = "NM1 (manual)", table(train_NM1$Class))
)
names(count_df) <- c("Scheme","Class","Freq")
ggplot(count_df, aes(x = Scheme, y = Freq, fill = Class)) +
  geom_col(position = "dodge", width = 0.7) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1)) +
  labs(title = "Class Counts Under Different Balancing Schemes", x = NULL, y = "Count")


## Grids & helper functions
clean_train_df <- function(df) {
  # drop rows with NA/Inf in any column
  df[] <- lapply(df, function(x) {
    if (is.numeric(x)) { x[!is.finite(x)] <- NA; x } else x
  })
  df <- df[complete.cases(df), , drop = FALSE]
  droplevels(df)
}



grid_glm   <- NULL
grid_rpart <- expand.grid(cp = c(0.001, 0.01))                       # small, effective
grid_nb    <- expand.grid(laplace=c(0,1), usekernel=c(TRUE,FALSE), adjust=1)
grid_knn   <- expand.grid(k = c(5, 17))
grid_svmR  <- expand.grid(sigma = 0.02, C = 1.0)
grid_nnet  <- expand.grid(size = 3, decay = 0.001)
p_try <- max(2, floor((ncol(train_scaled) - 1) / 3))
grid_rf    <- expand.grid(mtry = c(floor(sqrt(ncol(train_scaled)-1)), p_try))
#grid_xgb   <- expand.grid(
#  nrounds = c(100, 160), max_depth = c(3,5), eta = 0.1, gamma = 0,
#  colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)  ##kept for showing that we used
#parameter tuning to achieve the desired result.
grid_treebag <- NULL
RF_NTREES  <- 200
NNET_MAXIT <- 120
#Balanced XGBoost grid
grid_xgb_balanced <- expand.grid(
  nrounds = 90,          # a bit more boosting than 80
  max_depth = 2,          # shallow
  eta = 0.09,             # slightly larger than spec grid to help TPR_Yes
  gamma = 3,            # allow splits with moderate gain
  colsample_bytree = 0.7, # a touch richer features per tree than spec grid
  min_child_weight = 6,   # less strict than 5 -> recovers some TPR_Yes
  subsample = 0.7         # a bit higher than 0.7 -> stabilizes recall
)

#Compute required metrics on TEST at threshold = 0.5
compute_metrics <- function(obs, prob_yes) {
  pred <- factor(ifelse(prob_yes >= 0.5, "Yes", "No"), levels=c("Yes","No"))
  cm <- confusionMatrix(pred, obs, positive = "Yes")
  TP <- as.numeric(cm$table["Yes","Yes"])
  TN <- as.numeric(cm$table["No","No"])
  FP <- as.numeric(cm$table["Yes","No"])
  FN <- as.numeric(cm$table["No","Yes"])
  
  den <- sqrt( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) )
  num <- (TP * TN) - (FP * FN)
  
  mcc <- if (is.finite(den) && den > 0) num / den else 0
  tpr_yes <- TP/(TP+FN+1e-15); fpr_yes <- FP/(FP+TN+1e-15)
  prec_yes<- TP/(TP+FP+1e-15); f1_yes  <- ifelse((prec_yes+tpr_yes)==0,0, 2*prec_yes*tpr_yes/(prec_yes+tpr_yes))
  tpr_no  <- TN/(TN+FP+1e-15); fpr_no  <- FN/(FN+TP+1e-15)
  prec_no <- TN/(TN+FN+1e-15); f1_no   <- ifelse((prec_no+tpr_no)==0,0, 2*prec_no*tpr_no/(prec_no+tpr_no))
  supp_yes <- sum(obs=="Yes"); supp_no <- sum(obs=="No"); tot <- length(obs)
  wF1 <- (supp_yes/tot)*f1_yes + (supp_no/tot)*f1_no
  roc_auc <- tryCatch(pROC::roc(response=obs, predictor=prob_yes, levels=c("No","Yes"), quiet=TRUE)$auc %>% as.numeric(), error=function(e) NA_real_)
  
  kappa <- cm$overall["Kappa"][[1]]
  list(
    confusion = cm$table,
    metrics = data.frame(
      Measure=c("TPR","FPR","Precision","Recall","F1","ROC","MCC","Kappa"),
      Class_No  = c(tpr_no,fpr_no,prec_no,tpr_no,f1_no,roc_auc,mcc,kappa),
      Class_Yes = c(tpr_yes,fpr_yes,prec_yes,tpr_yes,f1_yes,roc_auc,mcc,kappa),
      Weighted_Avg = c(NA,NA,NA,NA,wF1,roc_auc,mcc,kappa)
    )
  )
}


# trainer 
train_and_eval <- function(train_df, test_df, method, tuneGrid=NULL, title="", tr_ctrl, ...) {
  y <- droplevels(train_df$Class)
  x <- train_df %>% dplyr::select(-Class)
  fit <- caret::train(
    x = x, y = y,
    method = method,
    metric = "Accuracy",          # CV selection by Accuracy
    trControl = tr_ctrl,
    tuneGrid = tuneGrid,
    preProcess = NULL,
    ...
  )
  prob <- tryCatch({
    predict(fit, newdata = test_df %>% dplyr::select(-Class), type = "prob")[,"Yes"]
  }, error=function(e){
    as.numeric(predict(fit, newdata = test_df %>% dplyr::select(-Class)) == "Yes")
  })
  res <- compute_metrics(test_df$Class, prob)
  safe <- function(s) gsub("[^A-Za-z0-9]+","_", s)
  write.csv(as.data.frame(res$confusion), paste0("CM_", safe(title), ".csv"))
  write.csv(res$metrics, paste0("METRICS_", safe(title), ".csv"), row.names = FALSE)
  list(model=fit, metrics=res$metrics, confusion=res$confusion, title=title)
}

#Tiny wrappers to pass speed args for rf/nnet where needed
fit_rf   <- function(train_df, ctrl, title, seeds)  fit_std(train_df, ctrl, "rf",   grid_rf,   title, seeds, ntree = RF_NTREES)
fit_nnet <- function(train_df, ctrl, title, seeds)  fit_std(train_df, ctrl, "nnet", grid_nnet, title, seeds, trace = FALSE, MaxNWts = 20000, maxit = NNET_MAXIT)
fit_std <- function(train_df, ctrl, method, grid, title, seeds, ...) {
  #1) Clean the training frame for any model
  train_df <- clean_train_df(train_df)
  
  #2) Rebuild folds to match the cleaned data
  ctrl2 <- ctrl
  ctrl2$index <- caret::createFolds(train_df$Class, k = 3, returnTrain = TRUE)
  ctrl2$indexOut <- lapply(ctrl2$index, function(tr) setdiff(seq_len(nrow(train_df)), tr))
  ctrl2$seeds <- seeds
  
  if (identical(method, "knn")) {
    #KNN path
    tr_xy <- make_numeric_xy(train_df)  # <- your helper that builds x (numeric matrix) + y + cols
    
    #Folds for knn must be built on this y to match tr_xy$x rows
    ctrl_knn <- ctrl2
    ctrl_knn$index <- caret::createFolds(tr_xy$y, k = 3, returnTrain = TRUE)
    ctrl_knn$indexOut <- lapply(ctrl_knn$index, function(tr) setdiff(seq_len(length(tr_xy$y)), tr))
    ctrl_knn$seeds <- seeds
    
    # Ensure no na.action interferes with knn3Train
    old_opts <- options(na.action = "na.pass")
    on.exit(options(old_opts), add = TRUE)
    
    if (anyNA(tr_xy$x)) stop("KNN train matrix still has NA after make_numeric_xy().")
    
    fit <- caret::train(
      x = tr_xy$x, y = tr_xy$y,
      method = "knn",
      metric = "Accuracy",
      trControl = ctrl_knn,
      tuneGrid = grid,
      preProcess = NULL,
      ...
    )
    
    te_xy <- make_numeric_xy(test_scaled, ref_cols = tr_xy$cols)
    if (anyNA(te_xy$x)) stop("KNN test matrix has NA after alignment.")
    
    prob <- tryCatch(
      predict(fit, newdata = te_xy$x, type = "prob")[, "Yes"],
      error = function(e) as.numeric(predict(fit, newdata = te_xy$x) == "Yes")
    )
    res <- compute_metrics(te_xy$y, prob)
    
  } else {
    #Generic path for all other models
    y <- droplevels(train_df$Class)
    x <- train_df %>% dplyr::select(-Class)
    
    if (anyNA(x)) stop("Training predictors contain NA after cleaning; check upstream preprocessing.")
    
    #GLM: make it stable under (quasi-)separation and reduce warnings
    if (identical(method, "glm")) {
      fit <- caret::train(
        x = x, y = y,
        method = "glm",
        metric = "Accuracy",
        trControl = ctrl2,
        tuneGrid = grid,
        family = binomial(link = "logit"),
        control = glm.control(maxit = 100, epsilon = 1e-08),
        ...
      )
    } else {
      fit <- caret::train(
        x = x, y = y,
        method = method,
        metric = "Accuracy",
        trControl = ctrl2,
        tuneGrid = grid,
        preProcess = NULL,
        ...
      )
    }
    
    prob <- tryCatch(
      predict(fit, newdata = test_scaled %>% dplyr::select(-Class), type = "prob")[, "Yes"],
      error = function(e) as.numeric(predict(fit, newdata = test_scaled %>% dplyr::select(-Class)) == "Yes")
    )
    res <- compute_metrics(test_scaled$Class, prob)
  }
  
  safe <- function(s) gsub("[^A-Za-z0-9]+","_", s)
  write.csv(as.data.frame(res$confusion), paste0("CM_", safe(title), ".csv"))
  write.csv(res$metrics, paste0("METRICS_", safe(title), ".csv"), row.names = FALSE)
  list(model = fit, metrics = res$metrics, confusion = res$confusion, title = title)
}

#36 EXPLICIT MODEL BUILDS
results <- list()
make_title <- function(balance, algo) paste0(balance, " + ", algo)

#1) DOWN (undersampling)
res_DOWN_GLM   <- fit_std(train_scaled, ctrl_down, "glm",        grid_glm,
                          make_title("Down","Logistic"),
                          seeds = make_caret_seeds(3L, n_grid_glm));      results <- append(results, list(res_DOWN_GLM))
res_DOWN_RPART <- fit_std(train_scaled, ctrl_down, "rpart",      grid_rpart,
                          make_title("Down","DecisionTree"),
                          seeds = make_caret_seeds(3L, n_grid_rpart));    results <- append(results, list(res_DOWN_RPART))
res_DOWN_NB    <- fit_std(train_scaled, ctrl_down, "naive_bayes",grid_nb,
                          make_title("Down","NaiveBayes"),
                          seeds = make_caret_seeds(3L, n_grid_nb));       results <- append(results, list(res_DOWN_NB))
res_DOWN_KNN   <- fit_std(train_scaled, ctrl_down, "knn",        grid_knn,
                          make_title("Down","KNN"),
                          seeds = make_caret_seeds(3L, n_grid_knn));      results <- append(results, list(res_DOWN_KNN))
res_DOWN_SVMR  <- fit_std(train_scaled, ctrl_down, "svmRadial",  grid_svmR,
                          make_title("Down","SVM_Radial"),
                          seeds = make_caret_seeds(3L, n_grid_svmR));     results <- append(results, list(res_DOWN_SVMR))
res_DOWN_NNET  <- fit_nnet(train_scaled, ctrl_down,               make_title("Down","NeuralNet"),
                           seeds = make_caret_seeds(3L, n_grid_nnet));     results <- append(results, list(res_DOWN_NNET))
res_DOWN_RF    <- fit_rf  (train_scaled, ctrl_down,               make_title("Down","RandomForest"),
                           seeds = make_caret_seeds(3L, n_grid_rf));       results <- append(results, list(res_DOWN_RF))
res_DOWN_XGB   <- fit_std(train_scaled, ctrl_down, "xgbTree",    grid_xgb_balanced,
                          make_title("Down","XGBoost"),
                          seeds = make_caret_seeds(3L, n_grid_xgbB),
                          nthread = 1, verbose = 0);                         results <- append(results, list(res_DOWN_XGB))
res_DOWN_BAG   <- fit_std(train_scaled, ctrl_down, "treebag",    NULL,
                          make_title("Down","Bagging"),
                          seeds = make_caret_seeds(3L, n_grid_bag));       results <- append(results, list(res_DOWN_BAG))

#2) UP (oversampling)
res_UP_GLM     <- fit_std(train_scaled, ctrl_up,   "glm",        grid_glm,
                          make_title("Up","Logistic"),
                          seeds = make_caret_seeds(3L, n_grid_glm));       results <- append(results, list(res_UP_GLM))
res_UP_RPART   <- fit_std(train_scaled, ctrl_up,   "rpart",      grid_rpart,
                          make_title("Up","DecisionTree"),
                          seeds = make_caret_seeds(3L, n_grid_rpart));     results <- append(results, list(res_UP_RPART))
res_UP_NB      <- fit_std(train_scaled, ctrl_up,   "naive_bayes",grid_nb,
                          make_title("Up","NaiveBayes"),
                          seeds = make_caret_seeds(3L, n_grid_nb));        results <- append(results, list(res_UP_NB))
res_UP_KNN     <- fit_std(train_scaled, ctrl_up,   "knn",        grid_knn,
                          make_title("Up","KNN"),
                          seeds = make_caret_seeds(3L, n_grid_knn));       results <- append(results, list(res_UP_KNN))
res_UP_SVMR    <- fit_std(train_scaled, ctrl_up,   "svmRadial",  grid_svmR,
                          make_title("Up","SVM_Radial"),
                          seeds = make_caret_seeds(3L, n_grid_svmR));      results <- append(results, list(res_UP_SVMR))
res_UP_NNET    <- fit_nnet(train_scaled, ctrl_up,                 make_title("Up","NeuralNet"),
                           seeds = make_caret_seeds(3L, n_grid_nnet));      results <- append(results, list(res_UP_NNET))
res_UP_RF      <- fit_rf  (train_scaled, ctrl_up,                 make_title("Up","RandomForest"),
                           seeds = make_caret_seeds(3L, n_grid_rf));        results <- append(results, list(res_UP_RF))
res_UP_XGB     <- fit_std(train_scaled, ctrl_up,   "xgbTree",    grid_xgb_balanced,
                          make_title("Up","XGBoost"),
                          seeds = make_caret_seeds(3L, n_grid_xgbR),
                          nthread = 1, verbose = 0);                         results <- append(results, list(res_UP_XGB))
res_UP_BAG     <- fit_std(train_scaled, ctrl_up,   "treebag",    NULL,
                          make_title("Up","Bagging"),
                          seeds = make_caret_seeds(3L, n_grid_bag));       results <- append(results, list(res_UP_BAG))

#3) SYN (manual synthetic oversampling; ctrl_none)
res_SYN_GLM    <- fit_std(train_SYN,     ctrl_none, "glm",        grid_glm,
                          make_title("SYN","Logistic"),
                          seeds = make_caret_seeds(3L, n_grid_glm));       results <- append(results, list(res_SYN_GLM))
res_SYN_RPART  <- fit_std(train_SYN,     ctrl_none, "rpart",      grid_rpart,
                          make_title("SYN","DecisionTree"),
                          seeds = make_caret_seeds(3L, n_grid_rpart));     results <- append(results, list(res_SYN_RPART))
res_SYN_NB     <- fit_std(train_SYN,     ctrl_none, "naive_bayes",grid_nb,
                          make_title("SYN","NaiveBayes"),
                          seeds = make_caret_seeds(3L, n_grid_nb));        results <- append(results, list(res_SYN_NB))
res_SYN_KNN    <- fit_std(train_SYN,     ctrl_none, "knn",        grid_knn,
                          make_title("SYN","KNN"),
                          seeds = make_caret_seeds(3L, n_grid_knn));       results <- append(results, list(res_SYN_KNN))
res_SYN_SVMR   <- fit_std(train_SYN,     ctrl_none, "svmRadial",  grid_svmR,
                          make_title("SYN","SVM_Radial"),
                          seeds = make_caret_seeds(3L, n_grid_svmR));      results <- append(results, list(res_SYN_SVMR))
res_SYN_NNET   <- fit_nnet(train_SYN,    ctrl_none,               make_title("SYN","NeuralNet"),
                           seeds = make_caret_seeds(3L, n_grid_nnet));      results <- append(results, list(res_SYN_NNET))
res_SYN_RF     <- fit_rf  (train_SYN,    ctrl_none,               make_title("SYN","RandomForest"),
                           seeds = make_caret_seeds(3L, n_grid_rf));        results <- append(results, list(res_SYN_RF))
res_SYN_XGB    <- fit_std(train_SYN,     ctrl_none, "xgbTree",    grid_xgb_balanced,
                          make_title("SYN","XGBoost"),
                          seeds = make_caret_seeds(3L, n_grid_xgbR),
                          nthread = 1, verbose = 0);                         results <- append(results, list(res_SYN_XGB))
res_SYN_BAG    <- fit_std(train_SYN,     ctrl_none, "treebag",    NULL,
                          make_title("SYN","Bagging"),
                          seeds = make_caret_seeds(3L, n_grid_bag));       results <- append(results, list(res_SYN_BAG))

na_rows <- train_NM1[!complete.cases(train_NM1), ]
# 4) NM1 (manual NearMiss-1 undersample; ctrl_none)
res_NM1_GLM    <- fit_std(train_NM1,     ctrl_none, "glm",        grid_glm,
                          make_title("NM1","Logistic"),
                          seeds = make_caret_seeds(3L, n_grid_glm));       results <- append(results, list(res_NM1_GLM))
res_NM1_RPART  <- fit_std(train_NM1,     ctrl_none, "rpart",      grid_rpart,
                          make_title("NM1","DecisionTree"),
                          seeds = make_caret_seeds(3L, n_grid_rpart));     results <- append(results, list(res_NM1_RPART))
res_NM1_NB     <- fit_std(train_NM1,     ctrl_none, "naive_bayes",grid_nb,
                          make_title("NM1","NaiveBayes"),
                          seeds = make_caret_seeds(3L, n_grid_nb));        results <- append(results, list(res_NM1_NB))
res_NM1_KNN    <- fit_std(train_NM1,     ctrl_none, "knn",        grid_knn,
                          make_title("NM1","KNN"),
                          seeds = make_caret_seeds(3L, n_grid_knn));       results <- append(results, list(res_NM1_KNN))
res_NM1_SVMR   <- fit_std(train_NM1,     ctrl_none, "svmRadial",  grid_svmR,
                          make_title("NM1","SVM_Radial"),
                          seeds = make_caret_seeds(3L, n_grid_svmR));      results <- append(results, list(res_NM1_SVMR))
res_NM1_NNET   <- fit_nnet(train_NM1,    ctrl_none,               make_title("NM1","NeuralNet"),
                           seeds = make_caret_seeds(3L, n_grid_nnet));      results <- append(results, list(res_NM1_NNET))
res_NM1_RF     <- fit_rf  (train_NM1,    ctrl_none,               make_title("NM1","RandomForest"),
                           seeds = make_caret_seeds(3L, n_grid_rf));        results <- append(results, list(res_NM1_RF))
res_NM1_XGB    <- fit_std(train_NM1,     ctrl_none, "xgbTree",    grid_xgb_balanced,
                          make_title("NM1","XGBoost"),
                          seeds = make_caret_seeds(3L, n_grid_xgbR),
                          nthread = 1, verbose = 0);                         results <- append(results, list(res_NM1_XGB))
res_NM1_BAG    <- fit_std(train_NM1,     ctrl_none, "treebag",    NULL,
                          make_title("NM1","Bagging"),
                          seeds = make_caret_seeds(3L, n_grid_bag));       results <- append(results, list(res_NM1_BAG))

# Leaderboard & Pick
stacked <- purrr::map_df(results, function(r) {
  m <- r$metrics
  data.frame(
    Model   = r$title,
    TPR_No  = m$Class_No[m$Measure=="TPR"],
    TPR_Yes = m$Class_Yes[m$Measure=="TPR"],
    F1_No   = m$Class_No[m$Measure=="F1"],
    F1_Yes  = m$Class_Yes[m$Measure=="F1"],
    ROC     = m$Class_Yes[m$Measure=="ROC"],
    MCC     = m$Class_Yes[m$Measure=="MCC"],
    Kappa   = m$Class_Yes[m$Measure=="Kappa"]
  )
})
write.csv(stacked, "ALL_MODELS_SUMMARY.csv", row.names = FALSE)

stacked <- stacked %>%
  mutate(MeetsMin = (TPR_Yes >= 0.72 & TPR_No >= 0.81),
         Score = 1.0*MeetsMin + 0.6*(TPR_Yes + TPR_No) + 0.4*ifelse(is.na(ROC), 0.5, ROC))
best_row <- stacked |>
  dplyr::arrange(dplyr::desc(Score)) |>
  dplyr::slice(1)

write.csv(best_row, "BEST_MODEL_SUMMARY.csv", row.names = FALSE)

print(best_row)

#PLOTS: Best Model Diagnostics
best_title <- as.character(best_row$Model[1])

# find the matching result list
best_obj <- NULL
for (r in results) {
  if (identical(r$title, best_title)) { best_obj <- r; break }
}
if (is.null(best_obj)) {
  message("Could not find best model object in `results`. Skipping plots.")
} else {
  # 6A) Confusion Matrix heatmap (test)
  cm_df <- as.data.frame(best_obj$confusion)
  names(cm_df) <- c("Predicted","Reference","Freq")  # caret prints table rows as Pred; cols as Ref
  ggplot(cm_df, aes(x = Reference, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", fontface = "bold") +
    scale_fill_gradient(low = "grey60", high = "black") +
    labs(title = paste("Confusion Matrix (Test) —", best_title),
         x = "Reference", y = "Predicted", fill = "Count")
  
  # 6B) ROC curve (use raw probs by re-predicting to be safe)
  best_fit <- best_obj$model
  testX <- test_scaled %>% dplyr::select(-Class)
  prob_best <- tryCatch(predict(best_fit, newdata = testX, type = "prob")[,"Yes"],
                        error = function(e) as.numeric(predict(best_fit, newdata = testX) == "Yes"))
  roc_obj <- pROC::roc(response = test_scaled$Class, predictor = prob_best,
                       levels = c("No","Yes"), quiet = TRUE)
  plot(roc_obj, main = paste("ROC (Test) —", best_title))
  
  # 6C) Precision–Recall curve with our own sweep
  thresh <- seq(0, 1, length.out = 201)
  pr_df <- lapply(thresh, function(t) {
    pred <- ifelse(prob_best >= t, "Yes", "No")
    TP <- sum(pred == "Yes" & test_scaled$Class == "Yes")
    FP <- sum(pred == "Yes" & test_scaled$Class == "No")
    FN <- sum(pred == "No"  & test_scaled$Class == "Yes")
    precision <- ifelse((TP+FP)==0, 1, TP/(TP+FP))
    recall    <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
    data.frame(threshold = t, precision = precision, recall = recall)
  }) %>% dplyr::bind_rows()
  
  ggplot(pr_df, aes(x = recall, y = precision)) +
    geom_path() +
    coord_cartesian(xlim = c(0,1), ylim = c(0,1)) +
    labs(title = paste("Precision–Recall Curve (Test) —", best_title),
         x = "Recall (TPR_Yes)", y = "Precision (Yes)")
  
  #6D) Threshold sweep for TPR_Yes and TPR_No vs threshold
  sweep_df <- lapply(thresh, function(t) {
    pred <- ifelse(prob_best >= t, "Yes", "No")
    TP <- sum(pred == "Yes" & test_scaled$Class == "Yes")
    TN <- sum(pred == "No"  & test_scaled$Class == "No")
    FP <- sum(pred == "Yes" & test_scaled$Class == "No")
    FN <- sum(pred == "No"  & test_scaled$Class == "Yes")
    tpr_yes <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
    tpr_no  <- ifelse((TN+FP)==0, 0, TN/(TN+FP))
    data.frame(threshold = t, TPR_Yes = tpr_yes, TPR_No = tpr_no)
  }) %>% dplyr::bind_rows() %>%
    tidyr::pivot_longer(cols = c("TPR_Yes","TPR_No"), names_to = "Metric", values_to = "TPR")
  
  ggplot(sweep_df, aes(x = threshold, y = TPR, color = Metric)) +
    geom_line() +
    labs(title = paste("Threshold Sweep —", best_title),
         x = "Threshold", y = "TPR")
  
  # 6E) Feature importance 
  suppressWarnings({
    vip <- try(caret::varImp(best_fit), silent = TRUE)
    if (!inherits(vip, "try-error")) {
      vip_df <- as.data.frame(vip$importance)
      vip_df$Feature <- rownames(vip$importance)
      vip_df <- vip_df %>%
        dplyr::mutate(Overall = if ("Overall" %in% names(.)) Overall else dplyr::first(dplyr::select(., dplyr::where(is.numeric)))) %>%
        dplyr::arrange(dplyr::desc(Overall)) %>%
        dplyr::slice(1:min(20, n()))
      ggplot(vip_df, aes(x = reorder(Feature, Overall), y = Overall)) +
        geom_col() +
        coord_flip() +
        labs(title = paste("Feature Importance —", best_title),
             x = NULL, y = "Importance")
    }
  })
}


top5 <- stacked[order(-stacked$Score), ][1:min(5, nrow(stacked)), ]
write.csv(top5, "TOP5_MODELS.csv", row.names = FALSE)

# Show top5 bar
top5_plot <- top5 %>%
  dplyr::mutate(Model = factor(Model, levels = rev(Model)))
ggplot(top5_plot, aes(x = Model, y = Score)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top Models by Composite Score",
       x = NULL, y = "Score")

