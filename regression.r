mldata = read.csv('ml_data.csv')

summary(lm(choice2 ~ choice1 + happen1, data=mldata))

summary(lm(choice2 ~ ML0 + ML1:happen1, data=mldata))

hist(mldata$ML1) # the predicted treatment effect

