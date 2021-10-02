library(dplyr)
library(magrittr)
library(tibble)
library(ggplot2)
library(assertthat)
library(stringr)
library(randomForest)
library(purrr)
library(xgboost)
library(pROC)
library(gtools)

FPATH <- 'data/hero_league_player_log.csv'
N_EXPECTED_HEROES <- 19
encode_winner <- function(s) {
  if (s == "Green Team Wins")
    1
  else if (s == "Blue Team Wins")
    0
  else
    stop("Unrecognized winner string")
}

dat <- read.csv(FPATH, stringsAsFactors = FALSE) %>% tibble::as_tibble()
all_heroes <- dat %>% select(-Result) %>% Reduce(c, .) %>% unique() %>% .[order(.)]
assert_that(are_equal(length(all_heroes), N_EXPECTED_HEROES))
columns <- all_heroes
row_to_onehot <- function(row) as.numeric(columns %in% row)
green_cols <- colnames(dat) %>% {.[str_detect(., "Green")]}
blue_cols <- colnames(dat) %>% {.[str_detect(., "Blue")]}

green_dat <- dat[green_cols]
blue_dat <- dat[blue_cols]
blues_mat <- apply(blue_dat, 1, row_to_onehot) %>% t()
greens_mat <- apply(green_dat, 1, row_to_onehot) %>% t()

greens_blues_mat <- cbind(greens_mat, blues_mat)

winners <- dat %>% 
  mutate(winner = case_when(Result == "Green Team Wins" ~ 1,
                            Result == "Blue Team Wins" ~ 0,
                            TRUE ~ NaN)) %$%
  winner
assert_that(!any(is.na(winners)))
indices <- list(trn = 1:55000, val = 55001:60000,
                tst = 60001:ncol(greens_blues_mat))
X <- list(trn=greens_blues_mat[indices$trn,],
          val=greens_blues_mat[indices$val,],
          tst=greens_blues_mat[indices$tst,])
y <- list(trn=winners[indices$trn],
          val=winners[indices$val],
          tst=winners[indices$tst])

max_depth = 4
max_nodes <- 2^(max_depth - 1)
params <- list(objective = "binary:logistic", eta = 0.05, alpha=3.5, lambda=0)
dtrn <- xgb.DMatrix(data = X$trn, label=y$trn)
dval <- xgb.DMatrix(data = X$val, label=y$val)

watchlist <- list(train=dtrn, val=dval)
model <- xgb.train(data=dtrn, watchlist=watchlist, nrounds=800,
                   params=params)

pred <- list()
pred$trn <- predict(model, X$trn)
pred$val <- predict(model, X$val)
pROC::auc(y$trn, pred$trn)
pROC::auc(y$val, pred$val)

opponent_team <- c("Dire Druid", "Greenery Giant", "Phoenix Paladin", 
                   "Quartz Questant", "Tidehollow Tyrant")
opponent_y <- sapply(columns, function(person) person %in% opponent_team) %>% as.numeric()
possible_our_teams_mat <- greens_mat
opponent_mat <- matrix(opponent_y, nrow=nrow(greens_mat), ncol=19, byrow=TRUE)
possible_games_mat <- cbind(possible_our_teams_mat, opponent_mat)
outcome_probas <- predict(model, possible_games_mat)
possible_our_teams_dat <- data.frame(possible_our_teams_mat) %>% as_tibble()
colnames(possible_our_teams_dat) <- columns
possible_our_teams_dat$win_proba <- outcome_probas
top_n <- 10
top_teams_dat <- possible_our_teams_dat %>% arrange(desc(win_proba)) %>% head(top_n)

permutations(n=19, r=5, v=columns, repeats.allowed = FALSE)

apply(top_teams_dat, 1, function(row) columns[c(row) == 1]) %>% t()
