library(Seurat)
library(data.table)
library(ape)
library(reshape2)
library(dplyr)
library(ggplot2)
library(stringr)

spatial_score <- function(seurat_obj) {
    # cluster just this seurat obj first
    seurat_obj <- NormalizeData(seurat_obj,
                       normalization.method = "LogNormalize",
                       scale.factor = 10000)
    seurat_obj <- FindVariableFeatures(seurat_obj,
                       selection.method = "vst",
                       nfeatures = 4000)
    seurat_obj <- ScaleData(seurat_obj,
                       features = rownames(seurat_obj))
    seurat_obj <- RunPCA(seurat_obj, npcs = 30)
    # seurat_obj <- RunUMAP(seurat_obj, dims = 1:30)
    # seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
    # seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)
    k <- 7 # expected layers
    seurat_obj@meta.data[, "kmeans_cluster"] <- kmeans(x = seurat_obj@reductions[["pca"]]@cell.embeddings,
            centers = k, nstart = 100)$cluster
    # entropy score
    kmeans_dist <- table(seurat_obj@meta.data$kmeans_cluster)
    kmeans_dist <- kmeans_dist / sum(kmeans_dist)
    entropy <- -sum(kmeans_dist * log(kmeans_dist))
    # neighborhood score
    kmeans_labels <- seurat_obj@meta.data[, c("orig.ident", "kmeans_cluster")]
    coord_table <- seurat_obj@images[[1]]@coordinates
    coord_table <- merge(coord_table, kmeans_labels, by=0)
    rownames(coord_table) <- coord_table$Row.names

    twoDmat <- acast(coord_table[, c("row", "col", "kmeans_cluster")], row ~ col, value.var="kmeans_cluster")
    tdm_dims <- dim(twoDmat)
    tdm_nrows <- tdm_dims[1]
    tdm_ncols <- tdm_dims[2]
    tdm_rows <- rownames(twoDmat)
    tdm_cols <- colnames(twoDmat)

    good_adjs <- c()
    bad_adjs <- c()
    for (i in 1:tdm_nrows) {
        for (j in 1:tdm_ncols) {
            if (!is.na(twoDmat[i,j])) {
                neighbor_idxs <- rbind(c(i-1, j-1), c(i-1, j+1),
                                       c(i, j-2), c(i, j+2),
                                       c(i+1, j-1), c(i+1, j+1))
                neighbor_idxs <- neighbor_idxs[neighbor_idxs[, 1] > 0, ]
                neighbor_idxs <- neighbor_idxs[neighbor_idxs[, 1] <= tdm_nrows, ]
                neighbor_idxs <- neighbor_idxs[neighbor_idxs[, 2] > 0, ]
                neighbor_idxs <- neighbor_idxs[neighbor_idxs[, 2] <= tdm_ncols, ]
                neighbor_clusts <- as.character(twoDmat[neighbor_idxs])
                
                for(nbr in neighbor_clusts[!is.na(neighbor_clusts)]) {
                    # note that you are double-counting each pair, but this should cancel out when taking a ratio of good/(good+bad) neighbors
                    if( nbr == twoDmat[i,j] ) {
                        good_adjs <- c(good_adjs, sprintf("%s %s", twoDmat[i,j], nbr))
                    } else {
                        bad_adjs <- c(bad_adjs, sprintf("%s %s", twoDmat[i,j], nbr))
                    }
                }
            }
        }
    }
    neighbor_score <- length(good_adjs) / (length(good_adjs) + length(bad_adjs))
    return(list(seurat_obj = seurat_obj, entropy = entropy, neighbor_score = neighbor_score))
}

set.seed(101)

# initial spatial data samples file
input_file_list <- "/data/neurogen/jy1008/scRNAseq_integration/all_data_QC_06252022/st_mtg_input_files_list_V4_10252023.csv"

useLogNorm <- TRUE
if(useLogNorm) {
    output_data_file <- "sample_qc_slot_lognorm_V4_10272023.xls"
    output_image_file <- "sample_qc_slot_lognorm_V4_output_10272023.pdf"
    output_st_sample_file <- "st_input_files_postqc_V4_10272023.csv"
    output_spatial_file <- "sample_qc_slot_lognorm_V4_singleclust_10272023.pdf"
} else {
    output_data_file <- "sample_qc_slot_counts_V4_10272023.xls"
    output_image_file <- "sample_qc_slot_counts_V4_output_10272023.pdf"
    output_st_sample_file <- "st_input_files_postqc_V4_10272023.csv"
    output_spatial_file <- "sample_qc_slot_counts_V4_singleclust_10272023.pdf"
}


# create the output table if it doesn't already exist
st_input_files <- read.csv(input_file_list, comment.char = '#')
# if (!file.exists(output_data_file)) {

# data.list <- list()
data.combined <- NULL
ii <- 1
entropies <- c()
neighbor_scores <- c()
pdf(output_spatial_file)
for (i in rownames(st_input_files)) {
   row_vals = st_input_files[i, ]
   if (row_vals$fold_removed) {
       print(paste(ii, row_vals$sample_name, "fold removed"), sep=', ')
       st_data <- readRDS(row_vals$dir)
       st_data <- subset(st_data, subset=selected_spot == FALSE)
   } else {
       print(paste(ii, row_vals$sample_name, "no fold removed"), sep=', ')
       st_data <- Load10X_Spatial(row_vals$dir)
   }
   st_data@meta.data["orig.ident"] <- row_vals$sample_name
   st_data@meta.data["batch"] <- row_vals$batch
   st_data@meta.data["sex"] <- row_vals$sex
   st_data@meta.data["expired_age"] <- row_vals$expired_age
   st_data@meta.data["diagnosis"] <- row_vals$diagnosis

   if(useLogNorm) {
       st_data <- NormalizeData(st_data, scale.factor = 10000)
       agg_counts <- AggregateExpression(st_data, group.by='orig.ident', slot='data', return.seurat=FALSE) # this exponentiates if slot='data'
   } else {
       agg_counts <- AggregateExpression(st_data, group.by='orig.ident', slot='counts', return.seurat=FALSE)
   }
   
   temp_res <- spatial_score(st_data)
   entropies <- c(entropies, temp_res$entropy)
   neighbor_scores <- c(neighbor_scores, temp_res$neighbor_score)
   st_pt <- SpatialDimPlot(temp_res$seurat_obj, group.by="kmeans_cluster") + ggtitle(paste(row_vals$sample_name, row_vals$diagnosis, format(round(temp_res$entropy, 4), nsmall = 4), format(round(temp_res$neighbor_score, 4), nsmall = 4)))
   print(st_pt)

   # df <- as.data.frame(agg_counts@assays$Spatial@counts)
   df <- as.data.frame(agg_counts$Spatial)
   colnames(df) <- row_vals$sample_name
   if (is.null(data.combined)) {
       data.combined <- df
   }
   else {
       data.combined <- merge(data.combined, df, by=0, all=TRUE)
       rownames(data.combined) <- data.combined$Row.names
       data.combined$Row.names <- NULL
   }
   # data.list[[ii]] = st_data
   ii <- ii + 1
}
dev.off()
st_input_files$entropy <- entropies
st_input_files$neighbor_score <- neighbor_scores

# combined <- merge(data.list[[1]], y = data.list[2:length(data.list)])
# combined$batch <- as.factor(combined$batch)
# combined$sex <- as.factor(combined$sex)
# combined@meta.data$expired_age <- droplevels(cut(combined@meta.data$expired_age, seq(0,130,by=5)))
# combined$diagnosis <- as.factor(combined$diagnosis)
# agg_counts <- AggregateExpression(combined, group.by='orig.ident', slot='counts', return.seurat=TRUE)
# write.table(agg_counts@assays$Spatial@counts, file=output_data_file, sep='\t', row.names=TRUE)
write.table(data.combined, file=output_data_file, sep='\t', row.names=TRUE)

# }
# st_input_files$batch <- as.factor(st_input_files$batch)
# st_input_files$sex <- as.factor(st_input_files$sex)
# st_input_files$expired_age <- droplevels(cut(st_input_files$expired_age, seq(0,130,by=5)))
# st_input_files$diagnosis <- as.factor(st_input_files$diagnosis)

fpkm=read.table(file(output_data_file), header=T, check.names =F)
scale_factor = 10000
fpkm <- sweep(fpkm,2,colSums(fpkm),`/`) * scale_factor
notAllZero <- (rowMeans(fpkm>0)>0.1)
logfpkm=fpkm[notAllZero,]
logfpkm=log10(logfpkm + 1e-4)  # so row value of 0 will be -4 in the transformed value
rle=logfpkm-apply(logfpkm, 1, median) # change "/" to "-" so that we got log(fold-change) which centered on 0 on the RLE plot.
rle=melt(cbind(ID=rownames(rle), rle), variable.name = "Sample",value.name ="FPKM", id="ID")

rle_summarize <- rle %>% group_by(Sample) %>% summarise(AbsMedian=abs(median(FPKM)))
rle_summarize <- as.data.frame(rle_summarize)
rle_summarize[order(rle_summarize$AbsMedian),]
st_input_files <- merge(st_input_files, rle_summarize, by.x="sample_name", by.y="Sample")
st_input_files$RLE_median_deviatn <- st_input_files$AbsMedian
st_input_files$AbsMedian <- NULL

pdf(output_image_file)
bymedian <- with(rle, reorder(Sample, FPKM, IQR))  # sort by IQR
op=par(mar=c(7,3,3,1))
boxplot(FPKM ~ bymedian, data=rle, outline=F, las=2, boxwex=1, col='gray', cex.axis=0.2, main="Relative Log Expression", xlab="", ylab="RLE", frame=F)
sampleDists = 1 - cor(fpkm, method='spearman')
hc=hclust(as.dist(sampleDists),method = "complete")

# To recolor according to covariates, we need to load in the combined Seurat object
# if it hasn't already been created here
# sample_covars <- subset(combined@meta.data, !duplicated(orig.ident),
#                         select=c("orig.ident", "batch", "sex", "expired_age", "diagnosis"))
# rownames(sample_covars) <- sample_covars$orig.ident


# celltype=gsub("(.*)_.*_(.*)_.*_.*","\\1_\\2",hc$labels)
# batch=gsub(".*_.*_.*_(\\d+)_.*","\\1",hc$labels)
# subject=gsub(".*_(.*)_.*_.*_.*","\\1",hc$labels)
# diagnosis <- sample_covars[hc$labels, "diagnosis"]
# batch <- sample_covars[hc$labels, "batch"]
# sex <- sample_covars[hc$labels, "sex"]
# subject <- hc$labels
rownames(st_input_files) <- st_input_files$sample_name
diagnosis <- st_input_files[hc$labels, "diagnosis"]
batch <- st_input_files[hc$labels, "batch"]
sex <- st_input_files[hc$labels, "sex"]
subject <- hc$labels


tree=as.phylo(hc)
## Update: fix edge.color bug. See https://stackoverflow.com/a/22102420/951718
myLabels <- c('node', sort(unique(batch)))
#myColors <- c("black", gray.colors(length(unique(batch)),start=0))
myColors <- c("black", rainbow(length(unique(batch))))
## match colors and labels (nomatch == node => select idx 1)
## (myLabels are reordered by edge ordering
batchColors <- myColors[match(batch[tree$edge[,2]], myLabels, nomatch=1)]

myColorsDiag <- rainbow(length(unique(diagnosis)))
diagColors <- myColorsDiag[diagnosis]

par(mar=c(1,1,1,1))
plot(tree, type = "unrooted",
     cex=.3, lab4ut='axial',underscore = T,
     tip.color=diagColors,
     edge.color= batchColors,
     main="Clustering of samples based on Spearman correlation")

# plot(tree, type = "unrooted",
#      cex=.3, lab4ut='axial',underscore = T,
#      tip.color=celltype.colors,
#      edge.color= batchColors,
#      main="Clustering of samples based on Spearman correlation")
# legend("bottomleft",
#        c("-- cell type --",unique(celltype),"-- batch --",paste("batch",sort(unique(batch)))),
#        text.col=c('black',paste0("#",cols$HEX[match(unique(celltype), cols$ITEM)]), myColors),
#        bty='n', cex=.5)

par(op)
D=apply(1-sampleDists, 1, median)
hist(D, breaks=100, ylab="Number of samples", xlab="D-statistic", main="Histogram of D-statistic")
cutoffD=quantile(D, probs = 0.05) # 5% quantitle
if(sum(D<cutoffD)) legend("topleft", paste(names(sort(D[which(D<cutoffD)])), round(sort(D[which(D<cutoffD)]),2)), bty='n', cex=.5)

sex_check <- t(fpkm[c("XIST", "RPS4Y1"), ])
sex_check <- as.data.frame(cbind(sex_check, sex=st_input_files[rownames(sex_check), "sex"]))
gp <- ggplot(sex_check, aes(x=XIST, y=RPS4Y1, color=sex)) + geom_point() + theme_bw()
print(gp)
dev.off()

st_input_files$d_stat <- D[st_input_files$sample_name]

write.table(st_input_files, file=output_st_sample_file, sep=',')
