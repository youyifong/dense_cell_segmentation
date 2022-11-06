# run these in the folders where the csi files are
    
library(kyotil)    
get_column_name <- function(name_list){
  filename <- c()
  for(j in 1:length(name_list)){
    name_list[j] = sub("_Position","",name_list[j])
    name_temp <- strsplit(name_list[j], split='_')
    name_temp <- paste(name_temp[[1]][2], name_temp[[1]][3], sep=' ')
    filename[j] = name_temp
  }
  return(filename)
}
get_avg_from_seeds <- function(file, header=T){
  res <- read.table(file, header=header, sep=',')
  res <- apply(res,2,mean)
  if (!header) names(res) = list.files("../testmasks")
  names(res) = get_column_name(names(res))
  # order in the following way
  ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
  res=res[order(match(names(res), ordered.names))]
  return(res)
}



# to run the experiment:
#   create training1-training7 folders, each containing the needed training images/masks files and testimages0-testimages2 folders
#   under images, run bash ../loop_cp_train_pred_eval.sh

for (i in 1:5) {
  labels=c("cyto","cyto2","tissuenet","livecell","none")
  
  # get AP
  names=c("AP_test_cyto","AP_test_cyto2","AP_test_tissuenet","AP_test_livecell","AP_test_none")
  files=paste0("csi_",labels[i],"_",ifelse(i==5,1,0):7,".txt")
  res=sapply(files, function(x) get_avg_from_seeds(x, header=T))
  colnames(res)=sub("csi_","",colnames(res))
  # print table
  res=t(rbind(res, mAP=colMeans(res))) 
  rownames(res)="Train"%.%0:(nrow(res)-1)
  colnames(res)=sub("L","",colnames(res))# remove L for lesion from names to be more succint
  #mytex(res, file=paste0("tables/",names[i]), digits=2, align="c")
  assign(names[i], res)
}

ylim=range(AP_test_cyto, AP_test_cyto2, AP_test_tissuenet, AP_test_livecell, AP_test_none)
k=ncol(AP_test_cyto)
training.size=c(423, 1450, 1082, 1620, 2255, 1458, 1818)
cum.training.size=c(0,cumsum(training.size))
names(training.size)=rownames(res)

mypdf(mfrow=c(2,3), file="AP_over_masks")
    mymatplot(cum.training.size, AP_test_cyto,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightblue",k-1),"blue"), cex=1, lty=c(2:k,1), main="Starting with Cyto",  y.intersp=.8)
    mymatplot(cum.training.size, AP_test_cyto2, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightgreen",k-1),"darkgreen"),       cex=1, lty=c(2:k,1), main="Starting with Cyto2", y.intersp=.8)
    mymatplot(cum.training.size, AP_test_tissuenet,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightpink",k-1),"lightpink3"), cex=1, lty=c(2:k,1), main="Starting with Tissuenet",  y.intersp=.8)
    mymatplot(cum.training.size, AP_test_livecell, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("goldenrod1",k-1),"goldenrod4"),       cex=1, lty=c(2:k,1), main="Starting with Livecell", y.intersp=.8)
    mymatplot(cum.training.size[-1], AP_test_none,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("mediumpurple1",k-1),"purple3"),       cex=1, lty=c(2:k,1), main="Starting with None", y.intersp=.8)
    mymatplot(cum.training.size, cbind(
        "Starting with cyto" =AP_test_cyto [,"mAP"], 
        "Starting with cyto2"=AP_test_cyto2[,"mAP"], 
        "Starting with tissuenet" =AP_test_tissuenet [,"mAP"], 
        "Starting with livecell"=AP_test_livecell[,"mAP"], 
        "Starting with none" =c(NA,AP_test_none [,"mAP"])),
      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","lightpink3","goldenrod4","purple3"), lty=1, pch=1, ylim=ylim, y.intersp=.8, type="b")
dev.off()



mypdf(mfrow=c(1,3), file="iAP_over_masks")
    mymatplot(cum.training.size, AP_test_cyto[,c(1:3,8)],  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col="blue", cex=1, lty=c(2:4,1), main="Starting with Cyto",  y.intersp=.8, pch=c(49:51,1))
    mymatplot(cum.training.size, AP_test_cyto[,c(4:7,8)],  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c("blue","blue","navy","navy","blue"), cex=1, lty=c(2:5,1), main="Starting with Cyto",  y.intersp=.8, pch=c(52:55,1))
    mymatplot(cum.training.size, cbind(
        "Starting with cyto" =AP_test_cyto [,"mAP"], 
        "Starting with cyto2"=AP_test_cyto2[,"mAP"], 
        "Starting with none" =c(NA,AP_test_none [,"mAP"])),
      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","purple3"), lty=1, pch=1, ylim=ylim, y.intersp=.8, type="b")
dev.off()


# print Bias results to tables
for (i in 1:3) {
    labels=c("cyto","cyto2","none")
    
    # get bias
    names=c("Bias_test_cyto","Bias_test_cyto2","Bias_test_none")
    files=paste0("bias_",labels[i],"_",ifelse(i==3,1,0):7,".txt")
    res=sapply(files, function(x) get_avg_from_seeds(x, header=F))
    colnames(res)=sub("bias_","",colnames(res))
    # print table
    res=t(rbind(res, Avg=colMeans(res))) 
    rownames(res)="Train"%.%0:(nrow(res)-1)
    colnames(res)=sub("L","",colnames(res))# remove L for lesion from names to be more succint
    mytex(res, file=paste0("tables/",names[i]), digits=2, align="c")
}


# print tp, fp, fn
res=read.csv("tpfpfn_cyto_2.txt")
#res=read.csv("tpfpfn_cyto_7.txt")

names(res) = sub("testimages_","",names(res))
names(res) = get_column_name(names(res))
ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
res=res[order(match(names(res), ordered.names))]

tab=sapply(res[1,], function(x) as.integer(strsplit(substr(x,2,nchar(x)-1)," +")[[1]]))
rownames(tab)=c("TP","FP","FN")
tab.1=rbind(Bias=formatDouble((tab["FP",]-tab["FN",])/(tab["TP",]+tab["FN",]),2),    
            AP=round(tab["TP",]/(tab["TP",]+tab["FP",]+tab["FN",]),2))
# normalize TP, FP, FN
gt=tab["TP",]+tab["FN",]
tab=rbind(formatDouble(t(t(tab)/gt),2, remove.leading0=F), tab.1)
colnames(tab)=sub("L","",colnames(tab))# remove L for lesion from names to be more succint
rownames(tab)[1:3]=rownames(tab)[1:3]%.%" prop"
tab

mytex(tab, file=paste0("tables/tpfpfn_cyto_2_seed0"), align="c")


###################################################################################################


# the new standard (std) differs from the regular in two aspects: no rotation, 448 patch size
# to get these results, first only do training, second only do prediction. In the second stage, change prediction parameters
files=c(
      "csi_std_flow4_cp-1.txt"  
    , "csi_std_flow4_cp1.txt"  
    , "csi_std_flow4_cp0.txt"  
    , "csi_std_flow3_cp0.txt"  
    , "csi_std_flow5_cp0.txt"  
)
res=sapply(files, function(x) get_avg_from_seeds(x))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("csi_","",colnames(res))
res=rbind(res, colMeans(res))
res

mytex(res, file=paste0("tables/AP_flow"), align="c")


# turning off individual random transformations has to be done in the cellpose python scripts
files=c(
    "csi_noscaling_448.txt", # set scale_range to 0 in core.py
    "csi_regular_448.txt", 
    "csi_noflip_448.txt", # set do_flip to False in random_rotate_and_resize()
    "csi_norotation_448.txt", # set theta to 0 in random_rotate_and_resize()
    "csi_nofliprotation_448.txt"
)
res=sapply(files, function(x) get_avg_from_seeds(x))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("csi_","",colnames(res))
res=rbind(res, colMeans(res))
res

mytex(res, file=paste0("tables/AP_data_augmentation"), align="c")


# patch size can be changed in command line
files=c(
    "csi_regular_56.txt",  
    "csi_regular_112.txt", 
    "csi_regular_224.txt", 
    "csi_regular_448.txt"
)
res=sapply(files, function(x) get_avg_from_seeds(x))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("csi_","",colnames(res))
res=rbind(res, colMeans(res))
res

mytex(res, file=paste0("tables/AP_patch"), align="c")


###################################################################################################
# collate results for deepcell training

tmp = list.files(path = "images/training/", pattern = "csi_tn1.0_nuclear_K_512x512resized_training.*.txt")
names(tmp)=1:7
out=sapply (tmp, function (f) {
    res <- read.table("images/training/"%.%f, header=T, sep=',')
    names(res) = get_column_name(names(res))
    #print(res)
    ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
    res=res[order(match(names(res), ordered.names))]
    #print(rowMeans(res)) # it shows that results from mpp 1.3 are always better than those from mpp 1
    res=unlist(res[2,])
    c(res, mAP=mean(res))
})
    
