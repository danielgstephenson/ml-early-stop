mldata = read.csv('ml_data.csv')
cleanData = read.csv('cleanData.csv')
options(scipen=99999)

mean(mldata$ML1)

summary(lm(choice2 ~ choice1 + happen1, data=mldata))

summary(lm(choice2 ~ ML0 + ML1:happen1, data=mldata))

ATE.hat = mean(mldata$ML1)
Z = ATE.hat*mldata$happen1
summary(lm(choice2 ~ ML0 + Z, data=mldata))

hist(mldata$ML1) # the predicted treatment effect

summary(lm(choice2~
             choice1 +
             happen1 +
             surveyTime +
             quizTime +
             approvals + 
             age,
           data=cleanData))

summary(lm(choice2~
             choice1 +
             happen1 +
             surveyTime:happen1 +
             quizTime:happen1 +
             approvals:happen1 + 
             age:happen1,
           data=cleanData))

summary(lm(choice2~
             choice1 +
             happen1 +
             surveyTime +
             surveyTime:happen1 +
             quizTime +
             quizTime:happen1 +
             approvals + 
             approvals:happen1 + 
             age +
             age:happen1,
           data=cleanData))

summary(lm(choice2~
             choice1 +
             happen1 +
             surveyTime +
             quizTime +
             approvals + 
             age +
             as.factor(ethnicity) +
             as.factor(nation) +
             as.factor(sex) +
             as.factor(employment) +
             as.factor(studentStatus) +
             as.factor(cognition1) +
             as.factor(cognition2),
           data=cleanData))

group0 = cleanData$happen1==0&cleanData$choice1==1
group1 = cleanData$happen1==1&cleanData$choice1==1
t.test(cleanData$choice2[group0],cleanData$choice2[group1])

