# helper functions
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
get_avg_from_seeds <- function(file){
  res <- read.table(file, header=T, sep=',')
  res <- apply(res,2,mean)
  #print(names(res))
  names(res) = get_column_name(names(res))
  # order in the following way
  ordered.names=c("JML8 CD8","JML8 CD3","JML8 CD4","JML9 CD3","JML10 CD3","CFL7 CD3","CFL13 CD3")
  res=res[order(match(names(res), ordered.names))]
  return(res)
}


# patch size can be changed in command line
files=c(
    "csi_regular_56.txt",  
    "csi_regular_112.txt", 
    "csi_regular_224.txt", 
    "csi_regular_448.txt"
)
res=sapply(files, function(x) get_avg_from_seeds(x))
res
colMeans(res)



files=c(
    "csi_noscaling_448.txt", # set scale_range to 0 in core.py
    "csi_regular_448.txt", 
    "csi_noflip_448.txt", # set do_flip to False in random_rotate_and_resize()
    "csi_norotation_448.txt" # set theta to 0 in random_rotate_and_resize()
)
res=sapply(files, function(x) get_avg_from_seeds(x))
res
colMeans(res)



############################################################
# Sunwoo

root <- '/Users/shan/Desktop/tmp/'
files <- paste(root, 'train', 1:5, '_ap_test_scratch.txt', sep='')
files <- c(paste(root, 'baseline_', 'ap_test_scratch.txt', sep=''), files) # for cyto and cyto2


res_mat <- matrix(NA, nrow=length(files), ncol=5) # 5 test images
for(i  in 1:length(files)){
  print(i)
  res_temp <- get_avg_from_seeds(file=files[i])
  res_mat[i,] <- res_temp[-5] # without P8 CD8 testtop
  if(i==1) {colnames(res_mat) <- get_column_name(name_list = names(res_temp)[-5])} # column name
}
res_mat <- res_mat[,c("M872956_Position8_CD8","M872956_Position8_CD3","M872956_Position8_CD4","M872956_Position9_CD3","M872956_Position10_CD3")]
rownames(res_mat) <- c('cyto2', paste('train', 1:5, sep='')) # for cyto and cyto2
#rownames(res_mat) <- paste('train', 1:5, sep='') # for scratch
colnames(res_mat) <- c("P8 CD8","P8 CD3","P8 CD4","P9 CD3","P10 CD3")
res_final <- cbind(res_mat, avg=apply(res_mat,1,mean))
AP_test_scratch <- res_final

save(AP_test_cyto, AP_train_cyto, AP_test_cyto2, AP_test_scratch, file='/Users/shan/Desktop/tmp/ap_results.RData')
load(file='/Users/shan/Desktop/tmp/ap_results.RData')
