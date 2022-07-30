library(kyotil)

load("data/ap_results.Rdata")

training.size=c(434,1489,1111,1638,2284)
names(training.size)=c("P8 CD8","P8 CD3","P8 CD4","P9 CD3","P10 CD3")
cum.training.size=c(0,cumsum(training.size))



###################################################################################################
# making graphs


ylim=range(AP_test_cyto, AP_test_cyto2, AP_test_scratch, AP_train_cyto)
k=nrow(AP_test_cyto2)

# comparing cyto with cyto2
myfigure(mfrow=c(1,3))
    mymatplot(cum.training.size, AP_test_cyto2, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightblue",k-1),"blue"), cex=1.5, lty=c(2:k,1), main="Starting with Cyto2", y.intersp=2)
    mymatplot(cum.training.size, AP_test_cyto, ylab="AP",  xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightgreen",k-1),"darkgreen"), cex=1.5, lty=c(2:k,1), main="Starting with Cyto", y.intersp=2)
    mymatplot(cum.training.size, cbind("Starting with cyto2"=AP_test_cyto2[,"avg"], "Starting with cyto"=AP_test_cyto[,"avg"]), ylab="AP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen"), lty=1, pch=1, ylim=ylim, y.intersp=2)
mydev.off(file="figures/cyto_cyto2_comparison")


# comparing train with test
myfigure(mfrow=c(1,3))
    mymatplot(cum.training.size[-1], AP_train_cyto[-1,], ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightblue",k-1),"blue"), cex=1.5, lty=c(2:k,1), main="Training", y.intersp=2)
    mymatplot(cum.training.size[-1], AP_test_cyto[-1,],  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightgreen",k-1), "darkgreen"),      cex=1.5, lty=c(2:k,1), main="Testing",  y.intersp=2)
    mymatplot(cum.training.size[-1], cbind("Training avg"=AP_train_cyto[-1,"avg"], "Testing avg"=AP_test_cyto[-1,"avg"]), ylab="AP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen"), lty=1, pch=1, ylim=ylim, y.intersp=2)
mydev.off(file="figures/test_train_comparison")
