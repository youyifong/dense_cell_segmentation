# run in dense_cell_segmentation
ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
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
# if each column corresponds to one test file
read_ap=function(file) {
  res <- read.table(file, header=T, sep=',')
  names(res) = get_column_name(names(res))
  res=res[order(match(names(res), ordered.names))]
  names(res)[names(res)=="NA NA"]="mAP"
  return(res)
}
get_avg_from_seeds <- function(file, header=T){
  res <- read.table(file, header=header, sep=',')
  res <- apply(res,2,mean)
  if (!header) names(res) = list.files("images/testmasks") # list.files returns sorted names, which matches how syotil checkprediction lists files as well 
  #print(names(res))
  names(res) = get_column_name(names(res))
  res=res[order(match(names(res), ordered.names))]
  return(res)
}


###################################################################################################
# summarize pretrained models results

# CellSeg
res=read.table("csi_cellseg.txt", header=T, sep=',')
names(res)=get_column_name (names(res))
res=res[order(match(names(res), ordered.names))]


# csi_cp_model_zoo.txt is created by cellpose_pred_model_zoo.sh
res_cp <- read.table("APresults/csi_cp_model_zoo.txt", header=T, sep=',')
rownames(res_cp)=res_cp[,1]
res_cp=res_cp[,-1]
names(res_cp) = get_column_name(names(res_cp))
res_cp=res_cp[order(match(names(res_cp), ordered.names))]
res_cp$mAP=rowMeans(res_cp)
res_cp


# csi_dc_model_zoo is edited by hand after created by DeepCell_predict_w_pretrained.ipynb
res_dc <- read.table("APresults/csi_dc_model_zoo.txt", header=T, sep=',')
rownames(res_dc)=res_dc[,1]
res_dc=res_dc[,-1]
names(res_dc) = get_column_name(names(res_dc))
res_dc=res_dc[order(match(names(res_dc), ordered.names))]
res_dc$mAP=rowMeans(res_dc)
rownames(res_dc)[2]="cytoplasm"
res_dc

# add deep distance models trained with tissuenet
res_dc=rbind(
    res_dc, 
    tn_nuclear=read_ap("APresults/csi_tn1.0_nuclear.txt"),
    tn_cyto=read_ap("APresults/csi_tn1.0_cyto.txt")
)
# remove mesmer and move cyto in front of nuclear
res_dc=res_dc[c(2,1,5,4),]
res_dc

# CellSeg
res_cs=read_ap("APresults/csi_cellseg.txt")
# compute mAP
res_cs=cbind(res_cs, mAP=rowMeans(res_cs))
# second row is best
res_cs=res_cs[2,,drop=F]
rownames(res_cs)=""
res_cs

res=rbind(res_cp, res_dc, res_cs); res

# write a table
res.1=res
colnames(res.1) = c(sapply(strsplit(ordered.names," "), function(x) x[2]),"")
mytex(res.1, file="tables/AP_pretrained", align="c", 
    col.headers="\\hline\n"%.%paste0("&",concatList(sapply(strsplit(ordered.names," "), function(x) x[1]),"&"),"&mAP")%.%"\\\\ ",
    add.to.row=list(list(0,nrow(res_cp),nrow(res_cp)+nrow(res_dc)), 
        c("       \n \\multicolumn{9}{l}{Cellpose} \\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{DeepCell}\\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{CellSeg}\\\\ \n"
         )
    )
)

# make a boxplot
myfigure(width=10,height=6)
    col=c(rep("black",nrow(res_cp)), rep("darkorange",nrow(res_dc)), rep("red",nrow(res_cs)))
    pch=c(rep(1,nrow(res_cp)), rep(2,nrow(res_dc)), rep(3,nrow(res_cs)))
    myboxplot(res, col=col, pch=pch, ylab="AP", cex=1.25)
    legend(x=4.85,y=.7,legend="Cellpose",pch=1,col=1,bty="n", pt.cex=1.25)
    legend(x=6,y=.7,legend="DeepCell",pch=2,col="darkorange",bty="n", pt.cex=1.25)
    legend(x=7.15,y=.7,legend="CellSeg", pch=3,col="red",bty="n", pt.cex=1.25)    
mydev.off(file="figures/AP_pretrained")



###################################################################################################
# training results vs mask

#### Cellpose models
# to run the experiment:
#   create training1-training7 folders, each containing the needed training images/masks files and testimages0-testimages2 folders
#   under images, run bash ../loop_cp_train_pred_eval.sh

labels=c("cyto","cyto2","cp_tissuenet","cp_livecell","cp_none")
names=c("AP_test_cyto","AP_test_cyto2","AP_test_tissuenet","AP_test_livecell","AP_test_none")
names(names)=labels
for (i in labels) {  
  files=paste0("APresults/csi_",i,"_",ifelse(i=="cp_none",1,0):7,".txt")
  res=sapply(files, function(x) get_avg_from_seeds(x, header=F))
  colnames(res)=sub("APresults/csi_","",colnames(res))
  res=t(rbind(res, mAP=colMeans(res))) 
  rownames(res)="Train"%.%ifelse(i=="cp_none",1,0):7
  # shorten image name - remove L for lesion from names to be more succint
  colnames(res)=sub("L","",colnames(res))
  #mytex(res, file=paste0("tables/",names[i]), digits=2, align="c")
  assign(names[i], res)
}

ylim=range(AP_test_cyto, AP_test_cyto2, AP_test_tissuenet, AP_test_livecell, AP_test_none)
k=ncol(AP_test_cyto)
training.size=c(423, 1450, 1082, 1620, 2255, 1458, 1818)
cum.training.size=c(0,cumsum(training.size))
names(training.size)=rownames(res)


# get results from DeepCell_tn_nuclear_K2a_series.ipynb and DeepCell_tn_cyto_K2a_series.ipynb
# train0
dc_ap=rbind(
    nuclear=read_ap("APresults/csi_tn1.0_nuclear.txt"),
    cyto=read_ap("APresults/csi_tn1.0_cyto.txt")
)
# train1-7
for(i in c("nuclear","cyto")) {
    tmp = list.files(path = "APresults", pattern = paste0("csi_tn1.0_", i, "_K_512x512resized_training.*.txt"))
    names(tmp)=1:7
    out=sapply (tmp, function (f) {
        res <- read.table("APresults/"%.%f, header=T, sep=',')
        names(res) = get_column_name(names(res))
        #print(res)
        ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
        res=res[order(match(names(res), ordered.names))]
        #print(rowMeans(res)) # it shows that results from mpp 1.3 are always better than those from mpp 1
        res=unlist(res[2,])
        c(res, mAP=mean(res))
    })
    out=t(out)
    rownames(out)=paste0("train", rownames(out))
    out=rbind(train0=dc_ap[i,], out)
    assign(paste0("AP_test_dc_",i), out)
}
AP_test_dc_nuclear
AP_test_dc_cyto

mAPs=cbind(
    "cyto" =AP_test_cyto [,"mAP"], 
    "cyto2"=AP_test_cyto2[,"mAP"], 
    "tissuenet" =AP_test_tissuenet [,"mAP"], 
    "livecell"=AP_test_livecell[,"mAP"], 
    "none" =c(NA,AP_test_none [,"mAP"]),
    "tn_nuclear" =AP_test_dc_nuclear [,"mAP"],
    "tn_cyto" =AP_test_dc_cyto [,"mAP"]
)
rownames(mAPs)[1]="Pretrained"

mytex(t(mAPs), file="tables/mAPs_over_masks", align="c", 
    add.to.row=list(list(0,5), 
        c("       \n \\multicolumn{9}{l}{Cellpose} \\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{DeepCell}\\\\ \n"
         )
    ))

myfigure(width=6, height=6)
    mymatplot(cum.training.size, mAPs,
      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","brown3","brown","purple3","goldenrod4","khaki4"), 
      lty=1, pch=1, ylim=c(.0,.75), y.intersp=.8, type="b", legend.x=8, legend.title="Initial Model", legend.cex=1)
mydev.off(file="figures/mAP_over_masks")


# make a boxplot
res=as.list(rbind(
    cyto=AP_test_cyto["Train7",], 
    cyto2=AP_test_cyto2["Train7",],
    tissuenet=AP_test_tissuenet["Train7",],
    livecell=AP_test_livecell["Train7",],
    none=AP_test_none["Train7",],
    tn_nuclear=AP_test_dc_nuclear["train7",],
    tn_cyto=AP_test_dc_cyto["train7",]
))

myfigure(width=10,height=6)
    col=c(rep("black",5), rep("goldenrod4",2))
    pch=c(1:5,6:7)
    myboxplot(res, col=col, pch=pch, ylab="AP", cex=1.25, boxwex=.8)
    legend(x=4.9-.5,y=.82,legend="cyto",pch=1,col="black",bty="n", pt.cex=1.25)
    legend(x=5.55-.5,y=.82,legend="cyto2",pch=2,col="black",bty="n", pt.cex=1.25)
    legend(x=6.3-.5,y=.82,legend="tissuenet", pch=3,col="black",bty="n", pt.cex=1.25)    
    legend(x=7.3-.5,y=.82,legend="livecell", pch=4,col="black",bty="n", pt.cex=1.25)    
    legend(x=8.1-.5,y=.82,legend="none", pch=5,col="black",bty="n", pt.cex=1.25)    
    legend(x=.85-.3,y=.35,legend="tn_nuclear",pch=7,col="goldenrod4",bty="n", pt.cex=1.25)
    legend(x=2-.3,y=.35,legend="tn_cyto",pch=7,col="goldenrod4",bty="n", pt.cex=1.25)
mydev.off(file="figures/AP_train7")



myfigure(mfrow=c(2,3))
    mymatplot(cum.training.size, AP_test_cyto,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightblue",k-1),"blue"), cex=1, lty=c(2:k,1), main="Starting with Cyto",  y.intersp=.8)
    mymatplot(cum.training.size, AP_test_cyto2, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightgreen",k-1),"darkgreen"),       cex=1, lty=c(2:k,1), main="Starting with Cyto2", y.intersp=.8)
    mymatplot(cum.training.size, AP_test_tissuenet,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightpink",k-1),"lightpink3"), cex=1, lty=c(2:k,1), main="Starting with Tissuenet",  y.intersp=.8)
    mymatplot(cum.training.size, AP_test_livecell, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("goldenrod1",k-1),"goldenrod4"),       cex=1, lty=c(2:k,1), main="Starting with Livecell", y.intersp=.8)
    mymatplot(cum.training.size[-1], AP_test_none,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("mediumpurple1",k-1),"purple3"),       cex=1, lty=c(2:k,1), main="Starting with None", y.intersp=.8)
    mymatplot(cum.training.size, mAPs,
      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","lightpink3","goldenrod4","purple3","brown3","brown"), 
      lty=1, pch=1, ylim=c(.0,.75), y.intersp=.8, type="b", legend.x=9, legend.title="Starting with", legend.cex=.8)    
mydev.off(file="figures/AP_over_masks")


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
# Optimization

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


####################
# deepcell data augmentation

ii=1:5
names(ii)=c("Regular", "NoScaling", "NoFlip", "NoRotation", "NoFlipRotation")
res=sapply (ii, function(i) {
    res <- read.table("APresults/csi_tn1.0_nuclear_K_512x512resized_train7_aug"%.%i%.%".txt", header=T, sep=',')
    names(res) = get_column_name(names(res))
    ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
    res=res[order(match(names(res), ordered.names))]
    res=unlist(res[2,])
    c(res, mAP=mean(res))
})

mytex(res, file=paste0("tables/AP_deepcell_augmentation"), align="c")
