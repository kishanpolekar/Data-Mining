library(e1071)
library(tree)
library(randomForest)

#Read data from test file
data=read.csv("segmentation test.txt", row.names=NULL)
#Set the first row name to Type
names(data)<-c("Type", names(data[2:20]))
data$Type=as.factor(data$Type)

#Print tabular format of data
fix(data)
dim(data)

attach(data)

#Perform naive bayes on entire dataset
nb=naiveBayes(Type~., data=data)
nb

#Predict naive bayes error rate for entire dataset
imagePredict=predict(nb,data)
imagePredict

#Stats for naive bayes on entire dataset
table(imagePredict,Type)
mean(imagePredict!=Type) #0.2004762 error rate

#Perform naive bayes and predict erorr rate for 1300 rows instead of entire dataset
set.seed(3)
train=sample(1:nrow(data),1300)
test=data[-train,]
test.label=Type[-train]
NB_2=naiveBayes(Type~.,data,subset=train)
NB_Predictions_2=predict(NB_2,test)
table(NB_Predictions_2,test.label)
mean(NB_Predictions_2!=test.label) #0.2125 error rate

#Perform naive bayes and predict erorr rate for 1800 rows instead of entire dataset
set.seed(3)
train=sample(1:nrow(data),1800)
test=data[-train,]
test.label=Type[-train]
NB_2=naiveBayes(Type~.,data,subset=train)
NB_Predictions_2=predict(NB_2,test)
table(NB_Predictions_2,test.label)
mean(NB_Predictions_2!=test.label) #0.1933333 error rate

#Perform naive bayes and predict erorr rate for 2000 rows instead of entire dataset
set.seed(3)
train=sample(1:nrow(data),2000)
test=data[-train,]
test.label=Type[-train]
NB_2=naiveBayes(Type~.,data,subset=train)
NB_Predictions_2=predict(NB_2,test)
table(NB_Predictions_2,test.label)
mean(NB_Predictions_2!=test.label) #0.15 error rate

#Decision tree based on our dataset and Type variable
tree.image=tree(Type~.,data)

#list the summary of the decision tree
summary(tree.image)

#predict the model for decision tree
tree.pred=predict(tree.image,type="class")

#Create the confusion matrix
table(tree.pred,Type)

#Correct prediction rate
mean(tree.pred==Type)

#Error prediction rate
mean(tree.pred!=Type) #error rate 0.0547619

#Random select a sample of 200 observations of the data set as a training set and the rest of the data set as a test set.
set.seed(3)
train=sample(1:nrow(data), 200)
data.test=data[-train,]
Type.test=Type[-train]
tree.image=tree(Type~.,data,subset=train)
tree.pred=predict(tree.image,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.09421053

#Cross validation testing
set.seed(3)
cv.data=cv.tree(tree.image,FUN=prune.misclass)
names(cv.data)
cv.data

plot(cv.data$size ,cv.data$dev ,type="b") #CV Bagging

#Pruning the bagging data
set.seed(3)
prune.data=prune.misclass(tree.image,best=9)
plot(prune.data)
text(prune.data,pretty=0)
tree.pred=predict(prune.data,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.09368421 ####################################### B E S T ###################################

#Pruning the bagging data #2
set.seed(3)
prune.data=prune.misclass(tree.image,best=5)
plot(prune.data)
text(prune.data,pretty=0)
tree.pred=predict(prune.data,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.3184211

#Pruning the bagging data #3
set.seed(3)
prune.data=prune.misclass(tree.image,best=11)
plot(prune.data)
text(prune.data,pretty=0)
tree.pred=predict(prune.data,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.09421053

#Random forest testing
set.seed(3)
tree.image=randomForest(Type~.,data,subset=train, ntree=500,mtry=10)
tree.pred=predict(tree.image,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #erorr rate 0.07315789

#Random forest testing #2
set.seed(3)
tree.image=randomForest(Type~.,data,subset=train, ntree=1500,mtry=10)
tree.pred=predict(tree.image,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.07315789

#Random forest testing #3
set.seed(3)
tree.image=randomForest(Type~.,data,subset=train, ntree=500,mtry=5)
tree.pred=predict(tree.image,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #error rate 0.07631579

#Random forest testing #4
set.seed(3)
tree.image=randomForest(Type~.,data,subset=train, ntree=500,mtry=13)
tree.pred=predict(tree.image,data.test,type="class")
table(tree.pred,Type.test)
mean(tree.pred!=Type.test) #erorr rate 0.07157895 ####################################### B E S T ###################################

#SVM Linear kernel tune best fit
set.seed (3)
tune.out=tune(svm,Type~.,data=data,kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1))) 

summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod) #best is cost 1; error rate 0.03761905

#SVM Linear kernel tune best fit #2
set.seed (3)
tune.out=tune(svm,Type~.,data=data,kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1,10))) 
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod) #best is cost 1; error rate 0.03761905

#SVM Linear kernel tune best fit #3
set.seed (3)
tune.out=tune(svm,Type~.,data=data,kernel="linear", ranges=list(cost=c(0.001,0.01,0.1,1,5))) 
summary(tune.out)
bestmod=tune.out$best.model
summary(bestmod) #best is cost 1; error rate 0.03761905

#SVM Radial kernel tune best fit 200 observations in training
set.seed (3)
tune.out=tune(svm,Type~.,data=data[train,],kernel="radial", ranges=list(cost=c(1,10,100), gamma=c(1,2,3)))
summary(tune.out) #all cases same with error 0.8
bestmod=tune.out$best.model
summary(bestmod) #will take first observation each time
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) #first obs table
mean(pred!=data[-train,"Type"]) #error rate 0.8615789

#SVM Radial kernel tune best fit 500 observations in training
set.seed (3)
train=sample(1:nrow(data), 500)
tune.out=tune(svm,Type~.,data=data[train,],kernel="radial", ranges=list(cost=c(1,10,100), gamma=c(1,2,3)))
summary(tune.out) #all cases same with error 0.872
bestmod=tune.out$best.model
summary(bestmod) #will take first observation each time
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) #first obs table
mean(pred!=data[-train,"Type"]) #error rate 0.860625

#SVM Radial kernel tune best fit 1300 observations in training
set.seed (3)
train=sample(1:nrow(data), 1300)
tune.out=tune(svm,Type~.,data=data[train,],kernel="radial", ranges=list(cost=c(1,10,100), gamma=c(1,2,3)))
summary(tune.out) #best case cost 10 gamma 1; error 0.8484615
bestmod=tune.out$best.model
summary(bestmod) #best case cost 10 gamma 1
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) 
mean(pred!=data[-train,"Type"]) #error rate 0.845 ####################################### B E S T ###################################

#SVM Polynomial kernel tune best fit 200 observations in training
set.seed (3)
tune.out=tune(svm,Type~.,data=data[train,],kernel="polynomial", ranges=list(cost=c(1,10,100), degree=c(1,2,3)))
summary(tune.out) #best is cost 10, degree 1; error rate 0.06
bestmod=tune.out$best.model
summary(bestmod) #best case cost 10 degree 1
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) 
mean(pred!=data[-train,"Type"]) #error rate 0.08473684

#SVM Polynomial kernel tune best fit 1600 observations in training
set.seed (3)
train=sample(1:nrow(data), 1600)
tune.out=tune(svm,Type~.,data=data[train,],kernel="polynomial", ranges=list(cost=c(1,10,100), degree=c(1,2,3)))
summary(tune.out) #best is cost 100, degree 1; error rate 0.04375
bestmod=tune.out$best.model
summary(bestmod) #best case cost 100 degree 1
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) 
mean(pred!=data[-train,"Type"]) #error rate 0.024

#SVM Polynomial kernel tune best fit 1800 observations in training
set.seed (3)
train=sample(1:nrow(data), 1800)
tune.out=tune(svm,Type~.,data=data[train,],kernel="polynomial", ranges=list(cost=c(1,10,100), degree=c(1,2,3)))
summary(tune.out) #best is cost 100, degree 1; error rate 0.04277778
bestmod=tune.out$best.model
summary(bestmod) #best case cost 100 degree 1
pred=predict(tune.out$best.model, newdata=data[-train,])
table(data[-train,"Type"], pred) 
mean(pred!=data[-train,"Type"]) #error rate 0.01666667 ####################################### B E S T ###################################
