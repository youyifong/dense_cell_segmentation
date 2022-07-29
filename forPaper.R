library(kyotil)


training.size=c(434,1489,1111,1638,2284)
training.size=cumsum(training.size)

###################################################################################################
# making graphs

# comparing cyto with cyto2 with four training images

AP.cyto2=c(
0.56, 0.54, 0.51, 0.44, 0.51,
0.62, 0.58, 0.50, 0.42, 0.53,
0.67, 0.70, 0.65, 0.55, 0.64,
0.66, 0.73, 0.65, 0.61, 0.66,
0.68, 0.78, 0.70, 0.63, 0.70)
AP.cyto2=matrix(AP.cyto2, ncol=5, byrow=T, dimnames=list(NULL, c("P8 CD8","P8 CD4","P8 CD3","P9 CD3","avg")))

AP.cyto=c(
0.64, 0.68, 0.55, 0.48, 0.59,
0.62, 0.59, 0.51, 0.42, 0.54,
0.68, 0.70, 0.64, 0.57, 0.65,
0.67, 0.73, 0.67, 0.62, 0.67,
0.69, 0.75, 0.69, 0.61, 0.69)
AP.cyto=matrix(AP.cyto, ncol=5, byrow=T, dimnames=list(NULL, c("P8 CD8","P8 CD4","P8 CD3","P9 CD3","avg")))

myfigure(mfrow=c(1,3))
    mymatplot(c(0,training.size[1:4]), AP.cyto2, ylab="AP", xlab="# of training masks", lwd=2, ylim=range(AP.cyto, AP.cyto2), col=c(rep("lightblue",4),"blue"), cex=1.5, lty=c(2,3,4,5,1), main="Starting with Cyto2", y.intersp=2)
    mymatplot(c(0,training.size[1:4]), AP.cyto, ylab="AP", xlab="# of training masks", lwd=2, ylim=range(AP.cyto, AP.cyto2), col=c(rep("lightgreen",4),"darkgreen"), cex=1.5, lty=c(2,3,4,5,1), main="Starting with Cyto", y.intersp=2)
    mymatplot(c(0,training.size[1:4]), cbind("Starting with cyto2"=AP.cyto2[,"avg"], "Starting with cyto"=AP.cyto[,"avg"]), ylab="AP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen"), lty=1, pch=1, ylim=range(AP.cyto, AP.cyto2), y.intersp=2)
mydev.off(file="figures/cyto_cyto2_comparison")


# comparing testing with training with five training images

AP.test=c(
0.61, 0.49, 0.55, 0.44, 0.39, 0.50,    #0.66, 
0.67, 0.65, 0.69, 0.57, 0.55, 0.62,    #0.78, 
0.65, 0.68, 0.73, 0.60, 0.57, 0.65,    #0.81, 
0.67, 0.69, 0.74, 0.61, 0.60, 0.66,    #0.80, 
0.71, 0.71, 0.77, 0.61, 0.64, 0.69)    #0.81, 
AP.test=matrix(AP.test, ncol=6, byrow=T, dimnames=list(NULL, c("P8 CD8","P8 CD3","P8 CD4","P9 CD3","P10 CD3","avg"))) # "P8 CD8 top",

AP.train=c(
0.67, 0.52, 0.44, 0.50, 0.39, 0.51,    #0.67, 
0.75, 0.67, 0.59, 0.65, 0.54, 0.63,    #0.72, 
0.77, 0.68, 0.68, 0.66, 0.56, 0.67,    #0.73, 
0.78, 0.69, 0.69, 0.70, 0.58, 0.69,    #0.74, 
0.77, 0.70, 0.70, 0.71, 0.63, 0.70)    #0.75, 
AP.train=matrix(AP.train, ncol=6, byrow=T, dimnames=list(NULL, c("P8 CD8","P8 CD3","P8 CD4","P9 CD3","P10 CD3","avg"))) # "P8 CD8 top",

myfigure(mfrow=c(1,3))
    mymatplot(training.size, AP.train, ylab="AP", xlab="# of training masks", lwd=2, ylim=range(AP.test, AP.train), col=c(rep("lightblue",5),"blue"), cex=1.5, lty=c(2:6,1), main="Training", y.intersp=2)
    mymatplot(training.size, AP.test,  ylab="AP", xlab="# of training masks", lwd=2, ylim=range(AP.test, AP.train), col=c(rep("lightgreen",5), "darkgreen"),      cex=1.5, lty=c(2:6,1), main="Testing",  y.intersp=2)
    mymatplot(training.size, cbind("Training avg"=AP.train[,"avg"], "Testing avg"=AP.test[,"avg"]), ylab="AP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen"), lty=1, pch=1, ylim=range(AP.test, AP.train), y.intersp=2)
mydev.off(file="figures/test_train_comparison")
