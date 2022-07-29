# helper functions
get_avg_from_seeds <- function(file){
  res <- read.table(file, header=T, sep=',')
  res <- apply(res,2,mean)
  return(res)
}

get_column_name <- function(name_list){
  filename <- c()
  for(j in 1:length(name_list)){
    name_temp <- strsplit(name_list[j], split='_')
    filename[j] <- paste(name_temp[[1]][1], name_temp[[1]][2], name_temp[[1]][3], sep='_')
  }
  return(filename)
}



# summary results
root <- '/Users/shan/Desktop/tmp/'
files <- paste(root, 'train', 1:5, '_ap_train_cyto.txt', sep='')
res_mat <- matrix(NA, nrow=length(files), ncol=5) # 6 test images
for(i  in 1:length(files)){
  print(i)
  res_temp <- get_avg_from_seeds(file=files[i])
  res_mat[i,] <- res_temp[-5] # without P8 CD8 testtop
  if(i==1) {colnames(res_mat) <- get_column_name(name_list = names(res_temp)[-5])} # column name
}
table13 <- res_mat
round(res_mat, 2)

save(table12, table13, table15, file='/Users/shan/Desktop/tmp/ap_results.RData')
load(file='/Users/shan/Desktop/tmp/ap_results.RData')
