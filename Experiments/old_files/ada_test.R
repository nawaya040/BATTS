library(ada)
library(caret)

# データの準備（前と同じ）
set.seed(123)
n <- 5000
p_data <- data.frame(x = rnorm(n, mean = 1), label = 1)
q_data <- data.frame(x = rnorm(n, mean = 0.5), label = 0)
train_data <- rbind(p_data, q_data)
train_data$label <- as.factor(train_data$label)

# チューニングのためのパラメータグリッド（iterを変化させる）
tune_grid <- expand.grid(iter = seq(100,1000, by = 100),
                         maxdepth = 4,   # 決定木の深さ（固定でもOK）
                         nu = 0.01)       # 学習率（固定でもOK）

# クロスバリデーション設定（5-fold CV）
ctrl <- trainControl(method = "cv", number = 5)

# モデル学習＆チューニング
set.seed(42)
model_cv <- train(label ~ ., data = train_data,
                  method = "ada",
                  trControl = ctrl,
                  tuneGrid = tune_grid,
                  metric = "Accuracy")  # 精度を最適化指標に

# 最適なパラメータと結果を表示
print(model_cv$bestTune)
plot(model_cv)


# second test
# 学習と検証データの分割（例: 8割学習, 2割検証）
set.seed(123)
id <- sample(1:nrow(dat), 0.8 * nrow(dat))
train <- dat[id, ]
valid <- dat[-id, ]

fit <- ada(y ~ ., data = train, iter = 160, control = rpart.control(maxdepth = 2), test.x = valid[, -1], test.y = valid$y)

# training error と test error
plot(1:160, fit$model$errs[,"train"], type = "l", col = "blue", ylim = c(0, 1), ylab = "Error", xlab = "Number of Trees")
lines(1:160, fit$model$errs[,"test"], col = "red")
legend("topright", legend = c("Train", "Test"), col = c("blue", "red"), lty = 1)

library(randomForest)
fit <- randomForest(Species ~ ., data = iris, importance = TRUE)
print(fit$err.rate)  # OOB error rateが出力される
