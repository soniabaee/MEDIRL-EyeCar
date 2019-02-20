library(mice)

pNeg1 <- function(x){sum(x == -1)/length(x)*100} ## prefer not to answer
pMiss <- function(x){(sum(is.na(x))/length(x))*100} ## NA value
aggr_plot <- function(data){ aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))}
imputeFunction_m5 <- function(data) {mice(data,m=5,maxit=5,meth='pmm',seed=245435)}

userEye = read.csv("/Users/soniabaee/Documents/University/Fall-2018/Human Factor/Project/eye_L_002.csv")
View(userEye)

userEye = userEye[grepl(pattern = "ET", userEye$EventSource) == TRUE,]


userEye$GazeX = round(apply(userEye[,c(grep("^GazeLeftx", colnames(userEye)),grep("^GazeRightx", colnames(userEye)))],1 ,mean))
userEye$GazeY = round(apply(userEye[,c(grep("^GazeLefty", colnames(userEye)),grep("^GazeRighty", colnames(userEye)))],1 ,mean))

userEye = userEye[,c('GazeX','GazeY','FixationSeq','FixationDuration')]

userEye = userEye %>% filter(!is.na(FixationDuration))

userEye[userEye == -1] = NA
apply(userEye, 2, pMiss)


md.pattern(userEye)


tempDataThreatData <- imputeFunction_m5(userEye)
imputedThreatData <- complete(tempDataThreatData,1)
apply(imputedThreatData, 2, pMiss)

View(userEye)





library(dplyr)
dim(userEye %>% filter(is.na(FixationDuration)))
names(userEye)
