library(mice)
library(dplyr)

pNeg1 <- function(x){sum(x == -1)/length(x)*100} ## prefer not to answer
pMiss <- function(x){(sum(is.na(x))/length(x))*100} ## NA value
aggr_plot <- function(data){ aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))}
imputeFunction_m5 <- function(data) {mice(data,m=5,maxit=10,meth='pmm',seed=245435)}

userData = read.csv("/Users/soniabaee/Documents/Projects/EyeCar/SensorData/usersData.csv")
intrClm = c('StudyName','ExportDate','Name', 'Age', 'Gender', 'StimulusName', 'SlideType', 'EventSource', 'Timestamp',
           'MediaTime', 'TimeSignal', 'GazeLeftx', 'GazeLefty', 'GazeRightx', 'GazeRighty', 'PupilLeft', 'PupilRight',
           'DistanceLeft', 'DistanceRight', 'CameraLeftX', 'CameraLeftY', 'CameraRightX', 'CameraRightY', 'ValidityLeft',
           'ValidityRight', 'GazeX', 'GazeY', 'GazeAOI', 'InterpolatedGazeX', 'InterpolatedGazeY', 'GazeEventType',
           'GazeVelocityAngle', 'SaccadeSeq', 'SaccadeStart', 'SaccadeDuration', 'FixationSeq', 'FixationX', 'FixationY',
           'FixationStart', 'FixationDuration', 'FixationAOI')
userData = userData[,intrClm]


users = unique(userData$Name)
data = data.frame()
for(u in users){
  userEye = userData %>% filter(Name == u)
  userEye = userEye[grepl(pattern = "ET", userEye$EventSource) == TRUE,]
  
  imputeClmn = c('InterpolatedGazeX', 'InterpolatedGazeY',
                'FixationSeq', 'FixationX','FixationY', 'FixationStart', 'FixationDuration')
  
  tmpdata = userEye[,imputeClmn]
  tmpdata[tmpdata == -1] = NA
  apply(tmpdata, 2, pMiss)
  tempData <- imputeFunction_m5(tmpdata)
  imputedData <- complete(tempData,1)
  tmpData2 = userEye[,!(names(userEye) %in% imputeClmn)]
  nData = cbind(imputedData, tmpData2)
  data = rbind(data,nData)
}
