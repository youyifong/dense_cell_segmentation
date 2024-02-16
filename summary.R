# run in dense_cell_segmentation
setwd("D:/DeepLearning/images_from_K/dense_cell_segmentation")    

# results area csi file are ordered according the following if there are no headers
default.file.order=unlist(strsplit(dir("images/test_gtmasks"), ".png")) 

{
# chronological order of the images
ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
numbered.names=c("1 CD8","2 CD3","3 CD4","4 CD3","5 CD3","6 CD3","7 CD3")
numbered.names.1=c("1_CD8","2_CD3","3_CD4","4_CD3","5_CD3","6_CD3","7_CD3")


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
  if (!header) names(res) = list.files("images/test_gtmasks") # list.files returns sorted names, which matches how syotil checkprediction lists files as well 
  #print(names(res))
  names(res) = get_column_name(names(res))
  res=res[order(match(names(res), ordered.names))]
  return(res)
}
get_map <- function(file, header=T){
  res <- read.table(file, header=header, sep=',')
  res <- apply(res,1,mean)
  res
}
}

###################################################################################################
# summarize pretrained models results
###################################################################################################


# csi_cp_model_zoo.txt is created by cellpose_pred_model_zoo.sh
res_cp <- read.table("APresults/cellpose/csi_cp_model_zoo.txt", header=T, sep=',')
rownames(res_cp)=res_cp[,1]
res_cp=res_cp[,-1]
names(res_cp) = get_column_name(names(res_cp))
res_cp=res_cp[order(match(names(res_cp), ordered.names))]
res_cp$mAP=rowMeans(res_cp)
res_cp


# csi_dc_model_zoo is edited by hand after created by DeepCell_predict_w_pretrained.ipynb
res_dc <- read.table("APresults/deepcell/csi_dc_model_zoo.txt", header=T, sep=',')
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
    tn_nuclear=read_ap("APresults/deepcell/csi_tn1.0_nuclear.txt"),
    tn_cyto=read_ap("APresults/deepcell/csi_tn1.0_cyto.txt")
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
rownames(res_cs)="CellSeg"
res_cs

# jacs
res_jacs=read.csv("APresults/pthmrcnn/csi_pthmrcnn_0.txt", header=F)
names(res_jacs)=get_column_name(default.file.order)
res_jacs=res_jacs[ordered.names]
res_jacs=cbind(res_jacs, mAP=rowMeans(res_jacs))
rownames(res_jacs)="jacs"
res_jacs


res=rbind(res_cp, res_dc, res_cs, res_jacs); res

# write a table
res.1=res
colnames(res.1) = c(sapply(strsplit(ordered.names," "), function(x) x[2]),"")
mytex(res.1, file="tables/AP_pretrained", align="c", 
    col.headers="\\hline\n"%.%paste0("&",concatList(sapply(strsplit(numbered.names," "), function(x) x[1]),"&"),"&mAP")%.%"\\\\ ",
    add.to.row=list(list(0,nrow(res_cp),nrow(res_cp)+nrow(res_dc)), 
        c("       \n \\multicolumn{9}{l}{Cellpose} \\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{DeepCell}\\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{Mask R-CNN}\\\\ \n"
         )
    )
)

# make a boxplot
res.1=res
colnames(res.1)=numbered.names.1
myfigure(width=10,height=6)
    col=c(rep("darkgreen",nrow(res_cp)), rep("goldenrod4",nrow(res_dc)), rep("tomato3",2))
    pch=c(rep(1,nrow(res_cp)), rep(6,nrow(res_dc)), rep(3,2))
    myboxplot(res.1, col=col, pch=pch, ylab="AP", cex=1.25)
    legend(x=4.85,y=.7,legend="Cellpose",pch=1,col="darkgreen",bty="n", pt.cex=1.25)
    legend(x=5.9,y=.7,legend="DC",pch=6,col="goldenrod4",bty="n", pt.cex=1.25)
    legend(x=6.7,y=.7,legend="MR-CNN", pch=3,col="tomato3",bty="n", pt.cex=1.25)    
mydev.off(file="figures/boxplot_AP_pretrained")




###################################################################################################
# training results as functions of number of masks
###################################################################################################

#### Cellpose models
# to run the experiment:
#   create training1-training7 folders, each containing the needed training images/masks files and testimages0-testimages2 folders
#   under images, run bash ../loop_cp_train_pred_eval.sh

labels=c("cp_cyto","cp_cyto2","cp_tissuenet","cp_livecell","cp_none")
names=c("AP_test_cyto","AP_test_cyto2","AP_test_tissuenet","AP_test_livecell","AP_test_none")
names(names)=labels
for (i in labels) {  
  files=paste0("APresults/cellpose/csi_",i,"_",ifelse(i=="cp_none",1,0):7,".txt")
  res=sapply(files, function(x) get_avg_from_seeds(x, header=F))
  colnames(res)=sub("APresults/cellpose/csi_","",colnames(res))
  res=t(rbind(res, mAP=colMeans(res))) 
  rownames(res)="Train"%.%ifelse(i=="cp_none",1,0):7
  # shorten image name - remove L for lesion from names to be more succint
  colnames(res)=sub("L","",colnames(res))
  #mytex(res, file=paste0("tables/",names[i]), digits=2, align="c")
  assign(names[i], res)
}

training.size=c(423, 1450, 1082, 1620, 2255, 1458, 1818)
cum.training.size=c(0,cumsum(training.size))
names(training.size)=rownames(res)


# save AP_test_cyto for addressing a revision comment
tab=t(AP_test_cyto)
colnames(tab)[1]="cyto"
tab
mytex(tab, file="tables/APs_over_masks_cyto", align="c")
# hide test values that are not out-of-sample
for(i in 1:7) tab[i,(i+1):8]=NA
tab
dat=t(tab)
colnames(dat)=c("1_CD8",  "2_CD3",  "3_CD4", "4_CD3",  "5_CD3", "6_CD3",  "7_CD3", "mAP")
{
myfigure(width=6, height=6)
mymatplot(cum.training.size[1:8], dat,
          ylab="AP", xlab="Fine-tuned models", lwd=2, col=1:7, legend.lty=NA,
          lty=1, pch=1:8, ylim=c(.0,.75), y.intersp=.5, type="b", legend.x=8, 
          legend.title="   Test Image", legend.cex=1,
          impute.missing.for.line=F,
          draw.x.axis = F, xaxt="n")
axis(1, at=cum.training.size, labels=F)
# adjust it so that Train1 will be drawn
cum.training.size1=cum.training.size
cum.training.size1[1]=-190
cum.training.size1[2]=700 
axis(1, at=cum.training.size1, labels=c("cyto","Train"%.%1:7), cex.axis=.75, tick=F)
mydev.off(file="figures/AP_over_masks_cyto")
}




ylim=range(AP_test_cyto, AP_test_cyto2, AP_test_tissuenet, AP_test_livecell, AP_test_none)
k=ncol(AP_test_cyto)


# get results from DeepCell_tn_nuclear_K2a_series.ipynb and DeepCell_tn_cyto_K2a_series.ipynb
# train0
dc_ap=rbind(
    nuclear=read_ap("APresults/deepcell/csi_tn1.0_nuclear.txt"),
    cyto=read_ap("APresults/deepcell/csi_tn1.0_cyto.txt")
)

#train1-7
for(i in c("nuclear","cyto")) {
    tmp = list.files(path = "APresults/deepcell", pattern = paste0("csi_tn1.0_", i, "_K_512x512resized_training.*.txt"))
    names(tmp)=1:7
    out=sapply (tmp, function (f) {
        res <- read.table("APresults/deepcell/"%.%f, header=T, sep=',')
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


# get results from pthmrcnn 
files="APresults/pthmrcnn/"%.%c(
      "csi_pthmrcnn_1.txt"  
    , "csi_pthmrcnn_2.txt"
    , "csi_pthmrcnn_3.txt"
    , "csi_pthmrcnn_4.txt"
    , "csi_pthmrcnn_5.txt"
    , "csi_pthmrcnn_6.txt"
    , "csi_pthmrcnn_7.txt"
)
AP_test_ptmrcnn=sapply(files, function(x) {
    res = read.csv(x, header=T)
    colMeans(res) # average aross seeds
})
colnames(AP_test_ptmrcnn)=1:7
rownames(AP_test_ptmrcnn)=get_column_name(rownames(AP_test_ptmrcnn))
AP_test_ptmrcnn=AP_test_ptmrcnn[ordered.names,]
AP_test_ptmrcnn=rbind(AP_test_ptmrcnn, mAP=colMeans(AP_test_ptmrcnn))
AP_test_ptmrcnn=t(AP_test_ptmrcnn)
AP_test_ptmrcnn

# combine
mAPs=cbind(
    "cyto" =AP_test_cyto [,"mAP"], 
    "cyto2"=AP_test_cyto2[,"mAP"], 
    "tissuenet" =AP_test_tissuenet [,"mAP"], 
    "livecell"=AP_test_livecell[,"mAP"], 
    "none" =c(NA,AP_test_none [,"mAP"]),
    "tn_nuclear" =AP_test_dc_nuclear [,"mAP"],
    "tn_cyto" =AP_test_dc_cyto [,"mAP"],
    "jacs" =c(.35, AP_test_ptmrcnn [,"mAP"])
)
rownames(mAPs)[1]="Pretrained"
tab=t(mAPs)
print(tab)

# write table
mytex(tab, file="tables/mAPs_over_masks", align="c", 
    add.to.row=list(list(0,5,7), 
        c("       \n \\multicolumn{9}{l}{Cellpose} \\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{DeepCell}\\\\ \n",
          "\\hline\n \\multicolumn{9}{l}{MR-CNN}\\\\ \n"
         )
))


# add ARI during revision
tab.1=cbind(tab
            , Train7=c(
              mean(get_avg_from_seeds("APresults/cellpose/ari_cyto.txt", header=F)),
              mean(get_avg_from_seeds("APresults/cellpose/ari_cyto2.txt", header=F)),
              mean(get_avg_from_seeds("APresults/cellpose/ari_tissuenet.txt", header=F)),
              mean(get_avg_from_seeds("APresults/cellpose/ari_livecell.txt", header=F)),
              mean(get_avg_from_seeds("APresults/cellpose/ari_none.txt", header=F)),
              mean(get_avg_from_seeds("APresults/deepcell/ari_tn1.0_nuclear_K_512x512resized_training7.txt", header=T)),
              mean(get_avg_from_seeds("APresults/deepcell/ari_tn1.0_cyto_K_512x512resized_training7.txt", header=T)),
              mean(get_avg_from_seeds("APresults/pthmrcnn/ari_pthmrcnn_7.txt", header=T))
            )
            # , Train7=c(
            #   mean(get_avg_from_seeds("APresults/cellpose/dice_cyto.txt", header=F)),
            #   mean(get_avg_from_seeds("APresults/cellpose/dice_cyto2.txt", header=F)),
            #   mean(get_avg_from_seeds("APresults/cellpose/dice_tissuenet.txt", header=F)),
            #   mean(get_avg_from_seeds("APresults/cellpose/dice_livecell.txt", header=F)),
            #   mean(get_avg_from_seeds("APresults/cellpose/dice_none.txt", header=F)),
            #   mean(get_avg_from_seeds("APresults/deepcell/dice_tn1.0_nuclear_K_512x512resized_training7.txt", header=T)),
            #   mean(get_avg_from_seeds("APresults/deepcell/dice_tn1.0_cyto_K_512x512resized_training7.txt", header=T)),
            #   mean(get_avg_from_seeds("APresults/pthmrcnn/dice_pthmrcnn_7.txt", header=T))
            # )
)
tab.1
mytex(tab.1, file="tables/mAPs_over_masks_ARI", align="c", 
      col.headers = "\\hline\n  &  \\multicolumn{8}{c}{mAP}  &  \\multicolumn{1}{c}{ARI}  \\\\  \n", 
      add.to.row=list(list(0,5,7), 
                      c("       \n \\multicolumn{10}{l}{Cellpose} \\\\ \n",
                        "\\hline\n \\multicolumn{10}{l}{DeepCell}\\\\ \n",
                        "\\hline\n \\multicolumn{10}{l}{MR-CNN}\\\\ \n"
                      )
      )
)



# make profile plot
mAPs.1=mAPs
colnames(mAPs.1)[1:5]="cp "%.%colnames(mAPs)[1:5]
colnames(mAPs.1)[6:7]="dc "%.%colnames(mAPs)[6:7]
colnames(mAPs.1)[8]="mrcnn "%.%colnames(mAPs)[8]
cols.1=c("olivedrab3","darkseagreen4","forestgreen","aquamarine4","darkgreen",   "orange3","goldenrod3",   "tomato3")
#    cols      =c("purple",          "green",              "olivedrab3",         "darkseagreen4",      "orange",    "cyan",           "tan",     "tomato3")
myfigure(width=6, height=6)
    mymatplot(cum.training.size, mAPs.1,
      ylab="mAP", xlab="# of training instances", lwd=2, col=cols.1, legend.lty=NA,
      lty=1, pch=1:8, ylim=c(.0,.75), y.intersp=.8, type="b", legend.x=8, legend.title="Initial Model", legend.cex=1)
mydev.off(file="figures/mAP_over_masks")


# rate of improvement after 4000 training instances
mean(mAPs["Train7",1:5]-mAPs["Train4",1:5])/(2255+1458+1818)*1000


# make boxplot
res=as.list(rbind(
    cyto=AP_test_cyto["Train7",], 
    cyto2=AP_test_cyto2["Train7",],
    tissuenet=AP_test_tissuenet["Train7",],
    livecell=AP_test_livecell["Train7",],
    none=AP_test_none["Train7",],
    tn_nuclear=AP_test_dc_nuclear["train7",],
    tn_cyto=AP_test_dc_cyto["train7",]
))

# draw boxplot
myfigure(width=10,height=6)
    pch=c(1:5,6:7)
    myboxplot(res, col=cols.1, pch=pch, ylab="AP", cex=1.25, boxwex=.8, ylim=c(.32,.8))
    legend(x=4.9-.6,y=.82,legend="cyto",pch=1,col=cols.1[1],bty="n", pt.cex=1.25)
    legend(x=5.55-.5,y=.82,legend="cyto2",pch=2,col=cols.1[2],bty="n", pt.cex=1.25)
    legend(x=6.3-.45,y=.82,legend="tissuenet", pch=3,col=cols.1[3],bty="n", pt.cex=1.25)    
    legend(x=7.3-.4,y=.82,legend="livecell", pch=4,col=cols.1[4],bty="n", pt.cex=1.25)    
    legend(x=8.1-.3,y=.82,legend="none", pch=5,col=cols.1[5],bty="n", pt.cex=1.25)    
    legend(x=.85-.3,y=.35,legend="tn_nuclear",pch=6,col=cols.1[6],bty="n", pt.cex=1.25)
    legend(x=2-.3,y=.35,legend="tn_cyto",pch=7,col=cols.1[7],bty="n", pt.cex=1.25)
mydev.off(file="figures/boxplot_AP_train7")



#myfigure(mfrow=c(2,3))
#    mymatplot(cum.training.size, AP_test_cyto,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightblue",k-1),"blue"), cex=1, lty=c(2:k,1), main="Starting with Cyto",  y.intersp=.8)
#    mymatplot(cum.training.size, AP_test_cyto2, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightgreen",k-1),"darkgreen"),       cex=1, lty=c(2:k,1), main="Starting with Cyto2", y.intersp=.8)
#    mymatplot(cum.training.size, AP_test_tissuenet,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("lightpink",k-1),"lightpink3"), cex=1, lty=c(2:k,1), main="Starting with Tissuenet",  y.intersp=.8)
#    mymatplot(cum.training.size, AP_test_livecell, ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("goldenrod1",k-1),"goldenrod4"),       cex=1, lty=c(2:k,1), main="Starting with Livecell", y.intersp=.8)
#    mymatplot(cum.training.size[-1], AP_test_none,  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c(rep("mediumpurple1",k-1),"purple3"),       cex=1, lty=c(2:k,1), main="Starting with None", y.intersp=.8)
#    mymatplot(cum.training.size, mAPs,
#      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","lightpink3","goldenrod4","purple3","brown3","brown"), 
#      lty=1, pch=1, ylim=c(.0,.75), y.intersp=.8, type="b", legend.x=9, legend.title="Starting with", legend.cex=.8)    
#mydev.off(file="figures/AP_over_masks")
#
#
#mypdf(mfrow=c(1,3), file="iAP_over_masks")
#    mymatplot(cum.training.size, AP_test_cyto[,c(1:3,8)],  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col="blue", cex=1, lty=c(2:4,1), main="Starting with Cyto",  y.intersp=.8, pch=c(49:51,1))
#    mymatplot(cum.training.size, AP_test_cyto[,c(4:7,8)],  ylab="AP", xlab="# of training masks", lwd=2, ylim=ylim, col=c("blue","blue","navy","navy","blue"), cex=1, lty=c(2:5,1), main="Starting with Cyto",  y.intersp=.8, pch=c(52:55,1))
#    mymatplot(cum.training.size, cbind(
#        "Starting with cyto" =AP_test_cyto [,"mAP"], 
#        "Starting with cyto2"=AP_test_cyto2[,"mAP"], 
#        "Starting with none" =c(NA,AP_test_none [,"mAP"])),
#      ylab="mAP", xlab="# of training masks", lwd=2, col=c("blue","darkgreen","purple3"), lty=1, pch=1, ylim=ylim, y.intersp=.8, type="b")
#dev.off()


## print Bias results to tables
#for (i in 1:3) {
#    labels=c("cyto","cyto2","none")
#    
#    # get bias
#    names=c("Bias_test_cyto","Bias_test_cyto2","Bias_test_none")
#    files=paste0("bias_",labels[i],"_",ifelse(i==3,1,0):7,".txt")
#    res=sapply(files, function(x) get_avg_from_seeds(x, header=F))
#    colnames(res)=sub("bias_","",colnames(res))
#    # print table
#    res=t(rbind(res, Avg=colMeans(res))) 
#    rownames(res)="Train"%.%0:(nrow(res)-1)
#    colnames(res)=sub("L","",colnames(res))# remove L for lesion from names to be more succint
#    mytex(res, file=paste0("tables/",names[i]), digits=2, align="c")
#}


# print tp, fp, fn
#res=read.csv("tpfpfn_cyto_2.txt")
##res=read.csv("tpfpfn_cyto_7.txt")
#
#names(res) = sub("testimages_","",names(res))
#names(res) = get_column_name(names(res))
#ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
#res=res[order(match(names(res), ordered.names))]
#
#tab=sapply(res[1,], function(x) as.integer(strsplit(substr(x,2,nchar(x)-1)," +")[[1]]))
#rownames(tab)=c("TP","FP","FN")
#tab.1=rbind(Bias=formatDouble((tab["FP",]-tab["FN",])/(tab["TP",]+tab["FN",]),2),    
#            AP=round(tab["TP",]/(tab["TP",]+tab["FP",]+tab["FN",]),2))
## normalize TP, FP, FN
#gt=tab["TP",]+tab["FN",]
#tab=rbind(formatDouble(t(t(tab)/gt),2, remove.leading0=F), tab.1)
#colnames(tab)=sub("L","",colnames(tab))# remove L for lesion from names to be more succint
#rownames(tab)[1:3]=rownames(tab)[1:3]%.%" prop"
#tab
#
#mytex(tab, file=paste0("tables/tpfpfn_cyto_2_seed0"), align="c")
#


###################################################################################################
# Cellpose optimization
###################################################################################################


############# patch size and data augmentation #################

# change patch size in cellpose_train_pred.sh
# change data augmentation options in cellpose_train_pred.sh and in python code
files="APresults/cellpose/"%.%c(
    "csi_cp_56.txt",  
    "csi_cp_112.txt", 
    "csi_cp_224.txt", 
    "csi_cp_448.txt",
#    "csi_cp_224_norotate.txt", # no need to show in table
    "csi_cp_448_norotate.txt",# set do_rotate to False in random_rotate_and_resize(). This is also exposed in main()
    "csi_cp_448_noflip.txt",# set do_flip to False in random_rotate_and_resize()
    "csi_cp_448_noscaling.txt" # set scale_range to 0 in core.py where it is passed to random_rotate_and_resize
)
res=sapply(files, function(x) get_avg_from_seeds(x))
res

t(sapply(files, function(x) get_map(x)))

colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("APresults/cellpose/csi_","",colnames(res))
res=rbind(res, mAP=colMeans(res))
res

range(res[,"cp_448_norotate"]-res[,"cp_448"])
range(res[,"cp_448_noflip"]-res[,"cp_448"])
range(res[,"cp_448_noscaling"]-res[,"cp_448"])
 
res.1=res
rownames(res.1)[1:7]=numbered.names.1
mytex(res.1, file=paste0("tables/AP_data_augmentation_cp"), align=c("c","c","c","c","c|","c","c","c"), include.colnames =F,       
    col.headers="\\hline\n 
         &\\multicolumn{1}{c}{56x56} &\\multicolumn{1}{c}{112x112} &\\multicolumn{1}{c}{224x224} &\\multicolumn{1}{c|}{448x448} & \\multicolumn{3}{c}{448x448} \\\\ 
         &\\multicolumn{4}{c|}{full data augmentation} & \\multicolumn{1}{c}{no rotate}& \\multicolumn{1}{c}{no flip} & \\multicolumn{1}{c}{no scale} \\\\ \\hline\n 
    "
)


################# Cellpose prediction parameters #################

files="APresults/cellpose/"%.%c(
      "csi_cp_448_norotate.txt"  
    , "csi_cp_448_norotate_flow3.txt"# modify cellpose_train_pred to get this
    , "csi_cp_448_norotate_flow5.txt"# when running the following, training part of cellpose_train_pred can be commented out
    , "csi_cp_448_norotate_cp-1.txt"  
    , "csi_cp_448_norotate_cp1.txt"  
)
res=sapply(files, function(x) get_avg_from_seeds(x))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("APresults/cellpose/csi_","",colnames(res))
res=rbind(res, mAP=colMeans(res))
res

res.1=res
rownames(res.1)[1:7]=numbered.names.1
mytex(res.1, file=paste0("tables/AP_prediction_param_cp"), align="c", include.colnames =F
    , col.headers="\\hline\n 
         &\\multicolumn{1}{c}{default} &\\multicolumn{2}{c}{flow threshold} &\\multicolumn{2}{c}{prob threshold} \\\\ 
         &\\multicolumn{1}{c}{} & \\multicolumn{1}{c}{0.3}& \\multicolumn{1}{c}{0.5} & \\multicolumn{1}{c}{-1} & \\multicolumn{1}{c}{1} \\\\ \\hline\n 
    "
)


# more seeds to compare with and without rotation
files=c("APresults/cellpose/csi_cp_448.txt", 
    "APresults/cellpose/csi_cp_448_more.txt") 
res=t(rbind(read.csv(files[1]), read.csv(files[2])))
files=c("APresults/cellpose/csi_cp_448_norotate.txt", 
    "APresults/cellpose/csi_cp_448_norotate_more.txt") 
res1=t(rbind(read.csv(files[1]), read.csv(files[2])))

mAP=colMeans(res)
mAP_norotate=colMeans(res1)
myboxplot(list(mAP, mAP_norotate))
abline(0,1)

wilcox.test(mAP, mAP_norotate, paired=T) # p value 0.03125
wilcox.test(mAP, mAP_norotate, paired=F) # p value 0.04113

cbind(mAP, mAP_norotate)
#           mAP mAP_norotate
#[1,] 0.6921246    0.7221790
#[2,] 0.6996127    0.7011673
#[3,] 0.7033981    0.7099636
#[4,] 0.7015911    0.7195936
#[5,] 0.7065550    0.7070350
#[6,] 0.7072379    0.7097623


# more tests at referee request

mAP=colMeans(read.csv("APresults/cellpose/csi_cp_448.txt"))
mAP_noflip=colMeans(read.csv("APresults/cellpose/csi_cp_448_noflip.txt"))
mAP_noscaling=colMeans(read.csv("APresults/cellpose/csi_cp_448_noscaling.txt"))

wilcox.test(mAP, mAP_noflip, paired=T) 
wilcox.test(mAP, mAP_noscaling, paired=T) 

cbind(mAP, mAP_noflip, mAP_noscaling)




# compare min size
files="APresults/cellpose/"%.%c(
      "csi_cp_448_norotate.txt"  
    , "csi_cp_448_norotate_minsize50.txt"
    , "csi_cp_448_norotate_minsize100.txt"
    , "csi_cp_448_norotate_minsize200.txt"
)
res=sapply(files, function(x) get_avg_from_seeds(x))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("APresults/cellpose/csi_","",colnames(res))
res=rbind(res, mAP=colMeans(res))
res

res.1=res
rownames(res.1)[1:7]=numbered.names.1
colnames(res.1)=c(0,50,100,200)
names(dimnames(res.1))=c("","min size")
mytex(res.1, file=paste0("tables/AP_min_size"), align="c", include.colnames =T)


###################################################################################################
# DeepCell data augmentation
###################################################################################################


ii=1:5
names(ii)=c("Regular", "NoScaling", "NoFlip", "NoRotation", "NoScalingRotation")
res=sapply (ii, function(i) {
    res <- read.table("APresults/deepcell/csi_tn1.0_nuclear_K_512x512resized_train7_aug"%.%i%.%".txt", header=T, sep=',')
    names(res) = get_column_name(names(res))
    ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
    res=res[order(match(names(res), ordered.names))]
    res=unlist(res[2,])
    c(res, mAP=mean(res))
})
res

mytex(res, file=paste0("tables/AP_data_augmentation_dc"), align="c")



################# tf mrcnn prediction #################

files="APresults/pthmrcnn/"%.%c(
      "csi_tfmrcnn1_060.txt"  
    , "csi_tfmrcnn1_080.txt"
    , "csi_tfmrcnn1_100.txt"
    , "csi_tfmrcnn1_120.txt"
    , "csi_tfmrcnn1_140.txt"
    , "csi_tfmrcnn1_160.txt"
    , "csi_tfmrcnn1_180.txt"
    , "csi_tfmrcnn1_200.txt"
    , "csi_tfmrcnn_cellseg.txt" # StringerEvalConfig
    , "csi_tfmrcnn_cellseg2.txt" # CVSegmentationConfig
)
res=sapply(files, function(x)   res <- unlist(read.csv(x, header=F)))
colnames(res)=sub(".txt","",colnames(res))
colnames(res)=sub("APresults/pthmrcnn/csi_","",colnames(res))
rownames(res)=default.file.order
res=rbind(res, mAP=colMeans(res))
res
