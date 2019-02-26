library(mice)

pNeg1 <- function(x){sum(x == -1)/length(x)*100} ## prefer not to answer
pMiss <- function(x){(sum(is.na(x))/length(x))*100} ## NA value
aggr_plot <- function(data){ aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))}
imputeFunction_m5 <- function(data) {mice(data,m=5,maxit=50,meth='pmm',seed=245435)}

#------------------------------------------------------------------------------------------------------
# 1. 30s == 450 frames, 1s ==> 15 frames
# 2. 30s == 3600 rows, 1s ==> 120 rows
# 1,2 ===> 8 rows == 1frame

#------------------------------------------------------------------------------------------------------
userEye = read.csv("C:/Users/Vishesh/Desktop/Sonia/Study_1_Group_1/Sensor Data/csv/001_Erfan_eyeData.csv")
View(userEye)

userEye = userEye[grepl(pattern = "ET", userEye$EventSource) == TRUE,]


userEye$GazeX = round(apply(userEye[,c(grep("^GazeLeftx", colnames(userEye)),grep("^GazeRightx", colnames(userEye)))],1 ,mean))
userEye$GazeY = round(apply(userEye[,c(grep("^GazeLefty", colnames(userEye)),grep("^GazeRighty", colnames(userEye)))],1 ,mean))

userEye = userEye[,c('Timestamp', 'FrameIndex', 'StimulusName', 'PupilRight', 'PupilLeft', 'DistanceLeft', 'DistanceRight', 'GazeX','GazeY','FixationSeq','FixationDuration')]


#------------------------------------------------------------------------------------------------------
## impute the data

slcImputeUser = userEye[,c('PupilRight', 'PupilLeft', 'DistanceLeft', 'DistanceRight', 'GazeX','GazeY')]

slcImputeUser[slcImputeUser == -1] = NA
apply(slcImputeUser, 2, pMiss)


md.pattern(slcImputeUser)


tempData <- imputeFunction_m5(slcImputeUser)
imputedData <- complete(tempData,1)
apply(imputedData, 2, pMiss)


nData = cbind(imputedData, userEye[,c('StimulusName', 'FixationSeq','FixationDuration')])

View(nData)

gazeFrame = data.frame(matrix(NA, nrow = dim(nData)[1], ncol = 10))
colnames(gazeFrame) = c('frame', 'stimulusName', 'pupilLeft', 'pupilRight', 'DistanceLeft', 'DistanceRight', 'GazeX', 'GazeY', 'FixationSeq','FixationDuration')
cnt = 1
for(stim in unique(nData$StimulusName)){
  stimData = nData %>% filter(StimulusName == 'Day_Sunny_High_1')
  for(i in seq(1, dim(stimData)[1]-8, 8)){
    j = i + 7
    tmpData = stimData[c(i:j), ]
    gazeFrame$frame[cnt] = (j/8)
    gazeFrame$stimulusName[cnt] = stim
    gazeFrame$pupilLeft[cnt] = mean(tmpData[["pupilLeft"]])
    gazeFrame$pupilRight[cnt] = mean(tmpData[["pupilRight"]])
    gazeFrame$DistanceLeft[cnt] = mean(tmpData[["DistanceLeft"]])
    gazeFrame$DistanceRight[cnt] = mean(tmpData[["GazeX"]])
    gazeFrame$GazeX[cnt] = mean(tmpData[["GazeY"]])
    gazeFrame$FixationSeq[cnt] = mean(tmpData[["FixationSeq"]], na.rm = TRUE)
    gazeFrame$FixationDuration[cnt] = mean(tmpData[["FixationDuration"]], na.rm = TRUE)
    cnt = cnt + 1 
  }
  
}





library(dplyr)
dim(userEye %>% filter(is.na(FixationDuration)))
names(userEye)
