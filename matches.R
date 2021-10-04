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
library(tidyr)

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

count_above_50 <- . %>%
  mutate(above_50 = win_proba >= 0.50) %>%
  group_by(above_50) %>%
  summarize(likely = n()) %>%
  mutate(total = sum(likely), prop = likely / total)


scale_y_percent <- list(
  scale_y_continuous(labels=partial(scales::percent, accuracy=1), limits=c(0, 1), breaks=seq(0, 1, 0.1)))
scale_x_percent <- list(
  scale_x_continuous(labels=partial(scales::percent, accuracy=1), limits=c(0, 1), breaks=seq(0, 1, 0.1)))

theme_percent <- list(theme(panel.grid.minor = element_blank(), axis.title = element_text(size=14), 
                            axis.text = element_text(size=12)))

dat <- read.csv(FPATH, stringsAsFactors = FALSE) %>% tibble::as_tibble()
set.seed(6)
dat %<>% {.[sample(nrow(.)),]}
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

################################################################################
## Train and evaluate predictor.
################################################################################
assert_that(!any(is.na(winners)))
indices <- list(trn = 1:55000, val = 55001:60000,
                tst = 60001:nrow(greens_blues_mat))
X <- list(trn=greens_blues_mat[indices$trn,],
          val=greens_blues_mat[indices$val,],
          tst=greens_blues_mat[indices$tst,])
y <- list(trn=winners[indices$trn],
          val=winners[indices$val],
          tst=winners[indices$tst])

params <- list(objective = "binary:logistic", eta = 0.03, lambda=4)
dtrn <- xgb.DMatrix(data = X$trn, label=y$trn)
dval <- xgb.DMatrix(data = X$val, label=y$val)
dall <- xgb.DMatrix(data = greens_blues_mat, label=winners)

watchlist <- list(train=dtrn, val=dval)
model_for_evaluation <- 
  xgb.train(data=dtrn, watchlist=watchlist, nrounds=350, params=params)
model <- xgb.train(dall, nrounds=350, params=params)


pred <- list()
pred$trn <- predict(model_for_evaluation, X$trn)
pred$val <- predict(model_for_evaluation, X$val)
pred$tst <- predict(model_for_evaluation, X$tst)
pred$all <- predict(model, greens_blues_mat)
pROC::auc(y$trn, pred$trn)
pROC::auc(y$val, pred$val)
pROC::auc(y$tst, pred$tst)
pROC::auc(winners, pred$all)

################################################################################
## Create team combinations.
################################################################################
opponent_team <- c("Dire Druid", "Greenery Giant", "Phoenix Paladin", 
                   "Quartz Questant", "Tidehollow Tyrant")
opponent_y <- sapply(columns, function(person) person %in% opponent_team) %>% as.numeric()
teams <- combinations(n=19, r=5, v=columns, repeats.allowed=FALSE)
possible_teams_mat <- apply(teams, 1, function(row) as.numeric(columns %in% row)) %>% t()
opponent_mat <- matrix(opponent_y, nrow=nrow(possible_teams_mat), ncol=19, byrow=TRUE)
possible_our_teams_mat <- possible_teams_mat
possible_games_mat <- cbind(possible_our_teams_mat, opponent_mat)
outcome_probas <- predict(model, possible_games_mat)
possible_our_teams_dat <- data.frame(possible_our_teams_mat) %>% as_tibble()
colnames(possible_our_teams_dat) <- columns
possible_our_teams_dat %<>% mutate(win_proba = outcome_probas) %>% arrange(desc(win_proba))

################################################################################
## Get best teams against this specific Blue team.
################################################################################
name_teams <- . %>%
  select(-win_proba) %>% 
  apply(1, function(row) columns[c(row) == 1]) %>% 
  t() %>% 
  as_tibble()
top_n <- 10
top_teams_dat <- possible_our_teams_dat %>% head(top_n)
top_probas <- possible_our_teams_dat %$% win_proba %>% head(top_n)
top_teams_named_dat <- top_teams_dat %>% name_teams() %>% mutate(win_proba = top_probas)
possible_our_teams_dat %>%
  as.data.frame() %>%
  {.[.[,"Arch-Alligator"] & .[,"Greenery Giant"] & .[,"Landslide Lord"] & 
    .[,"Nullifying Nightmare"] & .[,"Oil Ooze"],]}
colnames(top_teams_named_dat)[1:5] %<>% sapply(function (s) str_replace(s, "V", "person_"))
top_teams_named_dat

################################################################################
## Summarize character performance against this specific Blue team
################################################################################
possible_our_teams_dat %>% dplyr::mutate(win_proba_above_half = win_proba >= 0.5) %>%
  group_by(win_proba_above_half) %>%
  dplyr::summarize(likely = n()) %>%
  mutate(total = sum(likely), prop = likely / total)
  
possible_our_teams_dat %<>% dplyr::mutate(win_proba_above_cut = win_proba >= 0.6)
possible_our_teams_dat %>% group_by(win_proba_above_cut) %>% summarize(n())
above_cut_teams_dat <- possible_our_teams_dat %>% dplyr::filter(win_proba_above_cut)
person_prop_above_cut_dat <- above_cut_teams_dat %>% 
  .[columns] %>% 
  apply(2, mean) %>% 
  .[order(-.)] %>% 
  tibble(name=names(.), proba=.)
cat(person_prop_above_cut_dat %$% name %>% paste0(collapse="\n"))
cat(person_prop_above_cut_dat %$% proba %>% paste0(collapse="\n"))


# Only 1500 of 11500 possible teams give a >= 0.50 model score against the blue team.
# So they've got a pretty good team.
# Check whether these probabilities are well-calibrated.
all_win_probas_plot <- ggplot(possible_our_teams_dat) +
  geom_line(aes(x=1:nrow(possible_our_teams_dat), y=win_proba[order(-win_proba)])) +
  labs(x="Composition index", y="Win probability", 
       title="Win probabilities for all possible Green compositions,\nagainst the specified Blue team.") + 
  geom_hline(yintercept=0.5, color='blue', alpha=0.5) +
  scale_y_percent +
  scale_x_continuous(labels = function(x) paste0(x / 1000, "K"), breaks=seq(0, 12000, 1500)) +
  theme_bw() +
  theme_percent
ggsave(all_win_probas_plot, path="results", filename = "all_win_probas.png", dpi=170, width=6, height=3)

possible_our_teams_dat %>% count_above_50()
  

################################################################################
## Find out why Alligator is so good against this team. ########################
################################################################################
top_team_dat <- top_teams_dat %>% .[1,]
name_teams(top_team_dat)

alligator_teams_dat <- possible_our_teams_dat %>%
  as.data.frame() %>%
  .[.[,'Arch-Alligator'] == 1, columns] %>%
  as_tibble()

alligator_opponent_mat <- opponent_mat %>% .[1:nrow(alligator_teams_dat),]
alligator_games_mat <- cbind(as.matrix(alligator_teams_dat), alligator_opponent_mat)
alligator_preds <- predict(model, alligator_games_mat)
alligator_preds
################################################################################
## Get predicted win proportions over (a sample of) all possible games, 
## against all kinds of Blue teams.
################################################################################

make_crossjoinable <- function(mat, columns) {
  data.frame(mat) %>% set_colnames(columns) %>% mutate(k=TRUE)
}

crossjoin <- function(mat1, mat2, columns) {
  d1 <- make_crossjoinable(mat1, columns)
  d2 <- make_crossjoinable(mat2, columns)
  dplyr::inner_join(d1, d2, by='k', suffix=c(";green", ";blue")) %>% select(-k)
}

op_green_teams_mat <- possible_our_teams_mat %>%
  set_colnames(columns) %>%
  as.data.frame() %>%
  .[.[,"Nullifying Nightmare"] & .[,"Tidehollow Tyrant"] & .[,"Blaze Boy"] & .[,"Greenery Giant"],]
op_games_dat <- crossjoin(op_green_teams_mat, possible_our_teams_mat, columns)
op_games_pred <- predict(model, as.matrix(op_games_dat))
op_games_dat %<>% mutate(win_proba = op_games_pred)
op_win_probas_plot <- ggplot(op_games_dat) +
  geom_histogram(aes(x=win_proba), bins=60) +
  labs(x="Green win predicted probability", y = "# games",
       title="Win probabilities for matchups between OP Green teams\nand all possible Blue teams") +
  scale_x_percent +
  scale_y_continuous(labels=scales::comma) +
  geom_vline(xintercept=0.5, color='navy') +
  geom_vline(xintercept=mean(op_games_dat$win_proba), color='maroon') +
  theme_bw() +
  theme_percent +
  theme(panel.grid.minor = element_blank(), title = element_text(size=13))
height = 4
ggsave(op_win_probas_plot, path="results", filename="op_win_probas.png", dpi=140, width=height*2, height=height)
op_games_dat %>% count_above_50()
op_names_dat <- op_games_dat %>% 
  select(ends_with('green'), win_proba) %>% 
  rename_with(function(s) str_replace(s, ';green', ''))

op_names_dat %>%
  name_teams() %>%
  mutate(win_proba = op_names_dat$win_proba) %>%
  group_by(V1, V2, V3, V4, V5) %>%
  summarize(p = mean(win_proba)) %>%
  arrange(desc(p))

possible_blue_teams_mat <- possible_teams_mat
n_compare_teams <- 1400
set.seed(4)
subset_possible_blue_teams_mat <- possible_blue_teams_mat %>% 
  .[sample(nrow(.), n_compare_teams),]
subset_possible_green_teams_mat <- possible_our_teams_mat %>% 
  .[sample(nrow(.), n_compare_teams),]
dim(subset_possible_blue_teams_mat)

sample_games_dat <- crossjoin(
  subset_possible_green_teams_mat, 
  subset_possible_blue_teams_mat, columns) 
head(sample_games_dat, 3)
sample_games_pred <- predict(model, as.matrix(sample_games_dat))
sample_games_dat %<>% mutate(win_proba = sample_games_pred)

green_columns <- sapply(columns, . %>% paste0(";green"))
blue_columns <- sapply(columns, . %>% paste0(";blue"))

compute_mean_win_proba_for_opposing_pairs <- function(games_dat, name1, name2) {
  games_dat %>% 
    .[(.[,name1] == 1) & (.[,name2] == 1),] %>% 
    summarize(p = mean(win_proba), n = n()) %>%
    mutate(green_name=name1, blue_name=name2)
  
}

win_proba_nested_dats <- lapply(green_columns, 
       function(greencol) 
         lapply(blue_columns, function(bluecol)
           compute_mean_win_proba_for_opposing_pairs(sample_games_dat, greencol, bluecol)))

win_proba_dat <- lapply(win_proba_nested_dats, dplyr::bind_rows) %>% 
  dplyr::bind_rows() %>% 
  as_tibble() %>%
  mutate(p_dist = -(0.50 - p)) %>%
  tidyr::separate(green_name, c("green_name", "green_team"), ';') %>%
  tidyr::separate(blue_name, c("blue_name", "blue_team"), ';') %>%
  select(-green_team, -blue_team) %>%
  mutate(ratio_to_half = p / 0.5)

win_probas_tiles_plot <- win_proba_dat %>%
  ggplot(aes(x=green_name, y=blue_name, fill=p)) +
  geom_tile() +
  scale_fill_distiller(type = "div", palette = "RdYlBu", breaks = seq(0.3, 0.7, 0.05)) +
  labs(x="Green pick", y = "Blue\npick", 
       title = "How well do individual heroes do against each other?", 
       fill = "predicted probability\nof green win") +
  theme_bw() +
  theme(axis.text.x = element_text(angle=45, hjust=1, size=15),
        axis.text.y = element_text(size=15),
        axis.title.y = element_text(angle=0, size=15, vjust=0.5),
        axis.title.x = element_text(size=15),
        title = element_text(size=18),
        legend.title = element_text(size=15),
        legend.text = element_text(size=15),
        legend.key.height = unit(0.6, 'in'),
        legend.key.width = unit(0.4, 'in'),
        panel.grid = element_blank())
win_probas_tiles_plot
size <- 10
ggsave(win_probas_tiles_plot, path = "results", filename = "indiv_tiles.png", dpi=140, 
       width=size*1.3, height=size)

################################################################################
## Plotting. ###################################################################
################################################################################


buckets <- list(tst = round(pred$tst * 100) %/% 10 * 10)
buckets

bucket_means_dat <- tibble(bucket = buckets$tst, ytrue = y$tst, ypred = pred$tst) %>%
  group_by(bucket) %>%
  dplyr::summarize(true_win_prop = mean(ytrue), pred_win_prop = mean(ypred))
bucket_means_dat
calibration_plot <- bucket_means_dat %>%
  ggplot(aes(x=true_win_prop, y=pred_win_prop)) +
  geom_point(color='red', size=3) +
  geom_abline(size=1) +
  geom_line(color='red', size=1) +
  theme_bw() +
  scale_y_percent +
  scale_x_continuous(labels=partial(scales::percent, accuracy=1), limits=c(0, 1), breaks=seq(0, 1, 0.1)) +
  theme_percent +
  labs(x="true win %", y="predicted win %", title="Calibration")

ggsave(calibration_plot, path="results", filename = "calibration.png", dpi=170, width=6, height=3)
         