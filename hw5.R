# NB: Step 5: Drawing a Conclusion
# P-value <= significance level (a) => Reject your null hypothesis in favor of your alternative hypothesis.  Your result is statistically significant.
# P-value > significance level (a) => Fail to reject your null hypothesis.  Your result is not statistically significant.

# NB: Advice from piazza for 8.2:
# Using p-values to select model variables isn't recommended. For variable
# selection, use LASSO [least absolute shrinkage and selection operator]
# or stepwise regression (Efroymson 1960).
# If you want to see if a variable contributes to a model, you need to do a
# partial F-test. Refer to this link: What is a Partial F-Test? - Statology

# NB: Feature selection/reduction
# https://machinecreek.com/feature-subset-selection-in-r-using-embedded-approach/

# NB: can I try normalizing: data-mean(data)/sd(data).
# Still might give the same R2 though.

# NB: Pin: base model w/ all pred vars, then remove some predictors
# and IF SCALE, then UNSCALE before reporting answer

# qqplot, AIC (low better), cv.lm

# check for residual/leverage of outliers, remove if high in both

rm(list = ls())
setwd("/home/hank/github/algs/R/stats")

# R packages used:
library(pacman)
p_load(knitr, tidyverse, purrr, magrittr, assertthat,
       zeallot, rlist, readr, corrplot)
options(digits=2)
set.seed(1)

pacman::p_load(lintr, styler, prettycode,
                qcc, # CUSUM
                ggplot2,
                spc  # ARL (ave run length)
        )

# source("stats.R")

# Techniques: bollinger bands, ARL

sq <- function(x) x * x

set.seed(1)
options(digits = 4)
crimeResponseCol <- 16
tempYears <- 20

lg <- function(x) log(x, 2)
# for partitioning to train/validate/test
#c(cutoff1, cutoff2) %<-% c(CUTOFF_1, CUTOFF_2)

crimeMatrix <- read_tsv("data/uscrime.txt", show_col_types = F)

crimeCol <- crimeMatrix[, crimeResponseCol]
crimeVec <- crimeCol %>% as.matrix %>% as.vector
crimeVecSorted <- crimeVec %>% sort

# from OO:
uscrime <- read.table("data/uscrime.txt", stringsAsFactors=F, header=T)
crime <- uscrime[, "Crime"]
crime2 <- crime[-26]
crime3 <- crime2[-4]
#plot(uscrime[,16], type = "b")
#plot(uscrime$Pop,crime)
# Check normality:
# Null hypothesis: data is normal (so p small means reject null, data NOT normal?)
shapiro.test(crime)

crimeOutlierCandidateLocs <- c(4, 26, 27) #


# Question 6.2:
# CUSUM, ARL (average run length) instead of hyp test

tempMatrix <- read_tsv("data/temps.txt", show_col_types = F)
tempsDf <- as.data.frame(tempMatrix)

year2Temps <- function(year, df=tempsDf) {
  yearCol <- if (year < 30) year + 1 else year - 1994
  df[, c(1, yearCol)]
}

years2Cusums <- function(years, df=tempsDf) {
  tempsByYear <- lapply(1:20, year2Temps)
}

# CUSUM in ppt:
# https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm
# C_ is buffer, T is threshold
# INCREASES: S_t = max{0, S_t_1 + (x_t - mu - C)}, is S_t > T
# DECREASES: S_t = max{0, S_t_1 + (mu - x_t - C)}, is S_t > T
# inc = T when detecting increases, else inc = F detects decreases
cu_sum1d <- function(xs, C_ = 2, T_ = 0, inc = T, upper = T, scale = F) {
  if (scale) xs <- scale_(xs)
  N <- length(xs)
  mu <- mean(xs)

  ys <- if (inc) xs - mu - C_ else mu - xs - C_
  ys[1] <- 0

  Reduce(function(x, y) max(0, x + y), ys, accumulate=T)
}

y1 <- year2Temps(1996)[ ,2]
#c1 <- cu_sum1d(y1)
#c1 <- cusum(y1, decision.interval=6)
#c1b <- cusum(y1[80:100]) # decision.interval=5 by default

# yr <- 1996; cusum(year2Temps(yr)[,2][80:100])

# subrange can narrow date range, e.g. subrange = 80:100
# di = decision.interval
cusums <- function(years=1:20, di=6, subrange=NULL, plot_=F) {
  sr <- if (is.null(subrange)) 1:tempYears else subrange
  Map(
    function (i) cusum(year2Temps(i)[, 2][sr], decision.interval=di, plot=plot_),
    years)
}

# Use direction = "down" if crossing from above boundary to below boundary
crossingLocs <- function(xs, boundary = -decision_interval, direction="down") {
  N <- length(xs)

  inorder <- function(x, y, z)
    return(if (direction == "up") (x < y && y < z) else (x > y && y > z))

  crossingAcc <- function(i, acc) {
    if (i == N) return(acc)
    if (inorder(xs[i], boundary, xs[i+1])) acc <- append(i, acc)
    crossingAcc(i+1, acc)
  }
  crossingAcc(1, list()) %>% rev %>% unlist
}

decision_interval <- 6
cus <- Map(function(i) crossingLocs(cusum(year2Temps(i)[,2],
                  decision.interval=decision_interval, plot=F)$neg,
                  boundary = -decision_interval, direction = "down"),
                  1996:2015)

cus_len1 <- Filter(function(x) length(x) == 1, cus) %>% unlist
c(y4, y18) %<-% list(year2Temps(4)[, 2], year2Temps(18)[, 2])

cusAdjusted <- cus
cusAdjusted[[18]] <- 86
# boxplot(cusAdjusted)

f2 <- function(x) {
  if (x < 1/2) return(6*x) else return(-2*x + 4)
}

runningAve <- function(xss, f) {
  xs <- Map(f, xss) %>% unlist
  Map(function(i) mean(xs[1:i]), 1:length(xs)) %>% unlist
}

runningAveMax <- function(xss) runningAve(xss, max)
runningAveMin <- function(xss) runningAve(xss, min)

# counterexamples
zs <- c(seq(0, 64, 8), seq(64, 0, -1))
ws <- c(rep(34, 39), seq(34,0, -1))

ws2 <- ws

# exponential smoothing
simpleSmooth <- function(xs, alpha) {
  a <- alpha; b <- 1 - a
  Reduce(function(s, x) a*x + b*s, xs, accumulate = T)
}

simpleSmoothForLoop <- function(xs, alpha) {
  a <- alpha; b <- 1-a
  N <- length(xs)
  s <- rep(0, N)
  s[1] <- xs[1]
  for (i in 2:N) s[i] <- a*xs[i] + b*s[i - 1]
  return(s)
}

sinc_ <- function(x)
  #return(if (x == 0) 1 else sin(x) / x)
  ifelse(x == 0, 1, sin(x) / x)


xs <- seq(1,50, .2)
ys <- simpleSmooth(sinc_(xs), 0.15)
xys <- cbind(xs, ys)
xyDf <- xys %>% as.data.frame
xy <- xyDf
# plot(xys); curve(sinc_, add=T)

# Plotting, ggplot2, etc

mpg

plot1 <- ggplot(data = mpg) + geom_point(mapping = aes(x = displ, y = hwy))
plot1

gplot_mpg <- ggplot(data = mpg)
geom_pt <- geom_point(mapping = aes(x = displ, y = hwy))
# geom_smooth <- geom_smooth(mapping = aes(x = displ, y = hwy))

geomSmooth <- function(data, x, y)
  geom_smooth(mapping = aes(x = data$x, y = data$y))

gPlot <- function(type, data, x, y) {
    x <- enquo(x)
    y <- enquo(y)
    if (type == "smooth")
      ggplot(data=data) + geom_smooth(mapping = aes(x = !!x, y = !!y))
    else if (type == "point")
      ggplot(data=data) + geom_point(mapping = aes(x = !!x, y = !!y))
}

gplt_mpg_pt <- gplot_mpg + geom_pt
#gplt_mpg_smooth <- gplot_mpg + geom_smooth

# ggplot, stat_summary:
gPlot_statSummary <-
  ggplot(data = diamonds) +
    stat_summary(mapping = aes(x=cut, y=depth),
      fun.min = min, fun.max = max,
      fun = median
    )

ggPlotTemplate <- function(DATA, geom_f, aes_mapping) {
  ggplot(data = DATA) + geom_f(mapping = aes_mapping)
}

# NB: from OO for hw5, ~18:00
# R^2 is the amount of variance that the model is explaining

cor_crime <- cor(uscrime)
#corrplot(cor_crime)

lm_uscrime <- lm(Crime ~ ., data = uscrime)
plot(lm_uscrime)

boxplot(uscrime$Crime) # OO vid 20:36
out <- boxplot.stats(uscrime$Crime)$out
out

quantile <- function(x, xs) {
  loc <- 1
  for (y in xs) if (x > y) loc = loc + 1
  loc / (length(xs) + 1)
}

# Get quantiles of each df var relative to first N of df2 vars
quantiles <- function(df, df2, N) {
  options(digits=3)
  ns <- names(df)
  qs <- Map(function(i) quantile(df[i], df2[ ,i]), 1:N) %>% unlist
  cbind(ns, qs)
}


test_point <- data.frame(M = 14.0, So = 0, Ed = 10.0, Po1 = 12.0, Po2 = 15.5,
  LF = 0.640, M.F = 94.0, Pop = 150, NW = 1.1, U1 = 0.120, U2 = 3.6,
  Wealth = 3200, Ineq = 20.1, Prob = 0.040, Time = 39.0)

qs <- quantiles(test_point, uscrime, 15)

pred_model <- predict(lm_uscrime, test_point) # OO vid 21:34

lm_4 <- lm(Crime ~ Prob + Ineq + M + Ed, data = uscrime) # 22:57
summary(lm_4)

lm_6 <- lm(Crime ~ Prob + Ineq + M + Ed + Po1 + U2, data = uscrime) # 22:57

lm_5 <- lm(Crime ~ Po1 + Po2 + Pop + Wealth + Prob, data = uscrime)
pred_5 <- predict(lm_5, test_point)

lm_4b <- lm(Crime ~ Po1 + Pop + Wealth + Prob, data = uscrime)
pred_4b <- predict(lm_4b, test_point)

# lm_4c is lm5 without Wealth, since its coefficient was just -0.154
lm_4c <- lm(Crime ~ Po1 + Po2 + Pop + Prob, data = uscrime)
pred_4c <- predict(lm_4c, test_point)

# Cross validation, 30:00 min 00 vid
qqnorm(uscrime$Crime)
# library(DAAG) # cv.lm()
# lm_uscrime_cv <- cv.lm(uscrime, lm_4, m=4)
