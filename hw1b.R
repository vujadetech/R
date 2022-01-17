library(kernlab)
library(readr)
library(purrr)
library(comprehenr)

df <- as.matrix(read_tsv("data/credit_card_data-headers.txt"))
df <- df[1:300,]
DIR <- "csvs/vanilla/"
fullFname <- function(fname, DIR_=DIR) paste(DIR_, fname, sep="")

# Run ksvm on df_ with C and return (a, a0, pred), i.e., (weights, intercept, predictions)
C2weightsInterceptPreds <- function(C_, kernel_, df_=df) {
  # call ksvm.  Vanilladot is a simple linear kernel.
  # kernel_ 'arg' should be one of “rbfdot”, “polydot”, “tanhdot”, “vanilladot”, “laplacedot”, “besseldot”, “anovadot”, “splinedot”, “matrix”

  model <- ksvm(df_[,1:10],df_[,11],type="C-svc",kernel=kernel_,C=C_,scaled=T)

  # calculate a1…am
  a <- colSums(model@xmatrix[[1]] * model@coef[[1]])

  # calculate a0
  a0 <- -model@b

  # see what the model predicts
  pred <- predict(model, df_[,1:10])
  list(a, a0, pred) # return weights (a) and intercept (a0) as well in case needed for troubleshooting.
}

# xss is list of lists; return list consisting of xs[[i]] for each xs in xss:
proj_i <- function(xss, i) to_list(for (xs in xss) xs[[i]])

# Helper function for generating C values to try with ksvm, x^n for each n in ns:
powers <- function(x, ns) map(ns, function(n) x^n) %>% unlist

# Try numerous settings for parameter C at once: c(C1, C2, ...).
# Returns triple (weights, intercept, preds), so if success rate is all
# that is needed, use 'allSuccessRates' given below (and pass in model if computed to save time).
allCs2weightsInterceptPreds <- function(Cs=powers(10, 1:2), df_=df) # RETURNS VECTOR
  map(Cs, function(C_) C2weightsInterceptPreds(C_, df_))
# Can pass in models if already computed from previous function.
allSuccessRates <- function(Cs=powers(10, 1:2), models=NULL, fname) {
  # Compute models only if necessary:
  NUM_COL <- 2
  allModels <- if (is.null(models)) allCs2weightsInterceptPreds(Cs, df_=df) else models
  preds <- proj_i(allModels, 3) # prediction is in 3rd column
  ss <- map(preds, function (p) pred2successRate(p)) %>% unlist # ss=successes
  cs_ss <- list.zip(Cs %>% as.list, ss %>% as.list) %>% transpose
  m <- matrix(cs_ss %>% unlist, ncol=NUM_COL)
  if (!is.null(fname))
    write.table(m, fullFname(fname, DIR), row.names=F, col.names=F, sep=",")
  m
}

# Convert prediction to success rate by taking ratio:
# see what fraction of the model’s predictions match the actual classification
pred2successRate <- function(pred, df_=df) { sum(pred == df_[,11]) / nrow(df_) }

# GRAPHING TEMPLATE, e.g. ggTemplate(mpg, geom_point, aes(x=displ, y=hwy))
ggTemplate <- function(DATA, GEOM_FUNC, AES_MAPPINGS,
  POSITION="jitter") {
    ggplot(data = DATA) + GEOM_FUNC(mapping = AES_MAPPINGS, position=POSITION)
}
