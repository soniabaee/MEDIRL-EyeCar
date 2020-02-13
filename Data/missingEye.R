library(mice)

pNeg1 <- function(x){sum(x == -1)/length(x)*100} ## prefer not to answer
pMiss <- function(x){(sum(is.na(x))/length(x))*100} ## NA value
aggr_plot <- function(data){ aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))}
imputeFunction_m5 <- function(data) {mice(data,m=5,maxit=10,meth='pmm',seed=245435)}

#------------------------------------------------------------------------------------------------------
# 1. 30s == 450 frames, 1s ==> 15 frames
# 2. 30s == 3600 rows, 1s ==> 120 rows
# 1,2 ===> 8 rows == 1frame

#------------------------------------------------------------------------------------------------------
library(dplyr)
dataDir = "C:/Users/Vishesh/Desktop/Sonia/Study_2_Group_2/Sensor Data/csv/last/"
files = list.files(dataDir)

for(file in files){
  filDir = paste(dataDir,file,sep="")
  userEye = read.csv(filDir)
  
  
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
  
  #------------------------------------------------------------------------------------------------------
  ## create the framework 
  
  gazeFrame = data.frame(matrix(NA, nrow = dim(nData)[1], ncol = 10))
  colnames(gazeFrame) = c('frame', 'stimulusName', 'pupilLeft', 'pupilRight', 'DistanceLeft', 'DistanceRight', 'GazeX', 'GazeY', 'FixationSeq','FixationDuration')
  cnt = 1
  for(stim in unique(nData$StimulusName)){
    stimData = nData %>% filter(StimulusName == stim)
    for(i in seq(1, dim(stimData)[1]-8, 8)){
      j = i + 7
      tmpData = stimData[c(i:j), ]
      gazeFrame$frame[cnt] = (j/8)
      gazeFrame$stimulusName[cnt] = stim
      gazeFrame$pupilLeft[cnt] = mean(tmpData$PupilLeft)
      gazeFrame$pupilRight[cnt] = mean(tmpData$PupilRight)
      gazeFrame$DistanceLeft[cnt] = mean(tmpData$DistanceLeft)
      gazeFrame$DistanceRight[cnt] = mean(tmpData$DistanceRight)
      gazeFrame$GazeX[cnt] = mean(tmpData$GazeX)
      gazeFrame$GazeY[cnt] = mean(tmpData$GazeY)
      gazeFrame$FixationSeq[cnt] = mean(tmpData$FixationSeq, na.rm = TRUE)
      gazeFrame$FixationDuration[cnt] = mean(tmpData$FixationDuration, na.rm = TRUE)
      cnt = cnt + 1 
    }
    
  }
  
  saveDir = paste("C:/Users/Vishesh/Desktop/Sonia/Study_2_Group_2/Sensor Data/csv/",file,sep="" )
  write.csv(gazeFrame,saveDir )
  
}


#-----------------------------------------------------------
## hazardous value
aoiDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/AOI-data/"
aoifiles = list.files(aoiDir, "\\.csv$")

allAOI = data.frame()
for(file in aoifiles){
  filDir = paste(aoiDir,file,sep="")
  aoiData = read.csv(filDir)
  colName = aoiData[15,]
  aoiData = aoiData[-c(1:15),]
  colnames(aoiData) = as.character(unlist(colName[1,]))
  aoiData = aoiData[grepl(pattern = "^crash", aoiData$`AOI-Name`) == TRUE | grepl(pattern = "Crash", aoiData$`AOI-Name`) == TRUE | grepl(pattern = "Car-crash", aoiData$`AOI-Name`) == TRUE | grepl(pattern = "Car-cash", aoiData$`AOI-Name`) == TRUE ,]
  aoiData$Group = NULL
  aoiData$`Mouse Clicks` = NULL
  allAOI = rbind(aoiData,allAOI)
}

write.csv(allAOI,paste(aoiDir,"AOIData.csv",sep=""))

#-----------------------------------------------------------
## frameValue
frameDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/frame-data/"
framefiles = list.files(frameDir, "\\.csv$")

allFrame= data.frame()
for(file in framefiles){
  filDir = paste(frameDir,file,sep="")
  frameData = read.csv(filDir)
  frameData$study = unlist(strsplit(file,split='__', fixed=TRUE))[1]
  frameData$participant =  unlist(strsplit(unlist(strsplit(file,split='__', fixed=TRUE))[2],split='Frame', fixed=TRUE))[1]
  allFrame = rbind(frameData, allFrame)
}

allFrame = allFrame[rowSums(is.na(allFrame)) != 10,]


write.csv(allFrame,paste(frameDir,"FrameData.csv",sep=""))




#-----------------------------------------------------------
## read frame data
frameDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
frameData = read.csv(paste(frameDir,"FrameData.csv",sep=""))
frameData = frameData[rowSums(is.na(frameData)) != 10,]
View(frameData)



#-----------------------------------------------------------
## aoi value
aoiDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
aoiData = read.csv(paste(aoiDir, "AOIData.csv",sep=""))
aoiData = aoiData[,c("Study.name","Stimulus.Name" , "Respondent.Name" , "AOI.Total.Duration..ms.","Hit.time.G..ms." , "Time.spent.G..ms.", "TTFF.F..ms.", "Fixations.Count", "Average.Fixations.Duration..ms.")]
colnames(aoiData) = c("studyName", "stimulusName", "participant", "aoiDuration", "hitTime", "timeSpent", "ttff", "fixationCount", "avgFixDuration")
write.csv(aoiData,paste(aoiDir,"finalAOIData.csv",sep=""))
#-----------------------------------------------------------
## hazardous value
hazardousDir = "C:/Users/Vishesh/Desktop/Sonia/eyeCar-master/Data/InputData/"
hazardousData = read.csv(paste(hazardousDir, "hazardousFrame.csv",sep=""))
hazardousData = hazardousData[rowSums(is.na(hazardousData)) != 2,]
hazardousData$startFrame = hazardousData$startFrame*15
hazardousData$endFrame = hazardousData$endFrame*15





