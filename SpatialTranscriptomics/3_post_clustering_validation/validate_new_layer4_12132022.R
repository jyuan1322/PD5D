# pick the Leiden clustering resolution which produces the best layers

# - filter out small clusters
# - extract the cluster-specific marker genes using Presto
# - generate correlation w/ Maynard layers for each resolution
# - calculate the within-cluster distance sum for each resolution


library(Seurat)
library(SeuratWrappers)
library(ggplot2)
library(reshape2)
# library(gridExtra)
# load main clustering script
source("../spatial_clustering_pipeline_06252023/spatial_pipeline_filter_varfeat_duplicates_morecovars_05062022.R")

# updated for panda server
# seed 101, select res = 0.45
comb_seurat <- readRDS("/mnt/data0/projects/ASAP/ST/clustering_pipeline/all_data_5covar0p4_V4_seed101_10272023/comb_seurat_diet_out.rds")
comb_seurat@meta.data$num_label <- comb_seurat@meta.data$Spatial_snn_res.0.45
comb_seurat@meta.data$layer_label_v1 <- "Noise"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 7, "layer_label_v1"] <- "Layer 1"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 5, "layer_label_v1"] <- "Layer 2"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 2, "layer_label_v1"] <- "Layer 3"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 1, "layer_label_v1"] <- "Layer 5"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 4, "layer_label_v1"] <- "Layer 6"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 26, "layer_label_v1"] <- "WM_1"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 6, "layer_label_v1"] <- "WM_2"
comb_seurat@meta.data[comb_seurat@meta.data$num_label == 3, "layer_label_v1"] <- "WM_3"
comb_seurat@meta.data$num_label <- NULL
comb_seurat@meta.data$layer_label_v1 <- as.factor(comb_seurat@meta.data$layer_label_v1)
Idents(comb_seurat) <- comb_seurat@meta.data$layer_label_v1

# > table(comb_seurat@meta.data$layer_label_v1)
# Layer 1 Layer 2 Layer 3 Layer 5 Layer 6   Noise    WM_1    WM_2    WM_3
#   22336   53940   57502   69558   56347      63     891   22742   56951

layers_v1 <- sort(unique(comb_seurat@meta.data$layer_label_v1))
layers_v1 <- layers_v1[ layers_v1 != "Noise" ]
resolutions_list <- c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
harmony_vars <- c("orig.ident", "sex", "expired_age", "PMI", "RIN")
harmony_thetas <- c(0.4, 0.4, 0.4, 0.4, 0.4)
random_seed <- 101
for( layer in layers_v1 ) {
    print(paste("Subclustering", layer))
    comb_sub <- subset(comb_seurat, subset = layer_label_v1 == layer)
    job_name <- sprintf("seurat_subclust_5covar0p4_V4_%s_seed%s_10272023", gsub(' ', '', layer), random_seed)
    cluster_pipeline(comb_sub, job_name,
                     harmony_vars = harmony_vars,
                     harmony_thetas = harmony_thetas,
                     resolutions_list = resolutions_list,
                     random_seed = random_seed)
}


# 11/8/2023
# Based on visual inspection of the clusters mapped to sublayers, the following layers appear to have some spatially-significant sublayering:
# - Layer 1 (res=0.1)
# - Layer 5 extract Layer 4 (res=0.2)
# - Layer 5 remaining clusters (res=0.1)
# - WM_3 (res=0.1)


# extract Layer 4 from Layer 5; this is stored in layer_label_v2
layer <- "Layer 5"
subcluster_res <- 0.2
cluster_col <- paste0("Spatial_snn_res.", subcluster_res)
comb_sub <- readRDS(sprintf("/mnt/data0/projects/ASAP/ST/clustering_pipeline/post_clustering_validation_V4_seed101_10172023/seurat_subclust_5covar0p4_V4_%s_seed101_10272023/comb_seurat_diet_out.rds", gsub(' ','',layer)))
Idents(comb_sub) <- comb_sub@meta.data[, cluster_col]
comb_sub@meta.data[, "layer_label_v2"] <- as.character(comb_sub@meta.data[, cluster_col])
comb_sub@meta.data[comb_sub@meta.data$layer_label_v2 == 3, "layer_label_v2"] <- "Layer 4"
comb_sub@meta.data[comb_sub@meta.data$layer_label_v2 == 1, "layer_label_v2"] <- "Layer 5"
comb_sub@meta.data[comb_sub@meta.data$layer_label_v2 == 2, "layer_label_v2"] <- "Layer 5"
comb_sub@meta.data[comb_sub@meta.data$layer_label_v2 == 4, "layer_label_v2"] <- "Layer 5"
comb_sub@meta.data[comb_sub@meta.data$layer_label_v2 == 5, "layer_label_v2"] <- "Layer 5"

comb_seurat@meta.data$Row.names <- rownames(comb_seurat@meta.data)
comb_sub@meta.data$Row.names <- rownames(comb_sub@meta.data)
temp <- merge(comb_seurat@meta.data, comb_sub@meta.data[, c("Row.names", "layer_label_v2")], by="Row.names", all.x=TRUE)
rownames(temp) <- temp$Row.names
temp$Row.names <- NULL
temp <- temp[rownames(comb_seurat@meta.data), ]
temp[is.na(temp$layer_label_v2), "layer_label_v2"] <- as.character(temp[is.na(temp$layer_label_v2), "layer_label_v1"])
temp$layer_label_v2 <- as.factor(temp$layer_label_v2)
comb_seurat@meta.data <- temp



comb_seurat@meta.data$sublayer_label <- NA
subcluster_res <- 0.1




# color palette from:
# https://colorswall.com/palette/142
group.colors <- c("Layer 1"="#f43545", "Layer 2"="#ff8901", "Layer 3"="#fad717",
                  "Layer 4"="#00ba71", "Layer 5"="#00c2de", "Layer 6"="#ba55d3",
                  "WM_1"="#3b2175", "WM_2"="#75215b", "WM_3"="#652175", "Noise"="#c8c8c8")
# pdf("recolored_newlayer4_all_data_5covars0p4_seed101_res0p45_12142022.pdf")
pdf("recolored_layers_plusnewL4_unsmoothed_V4_11082023.pdf")
for (img in names(comb_seurat@images)) {
    meta_row <- rownames(comb_seurat@images[[img]]@coordinates)[1]
    sample_id <- comb_seurat@meta.data[meta_row, "orig.ident"]
    diagnosis <- comb_seurat@meta.data[meta_row, "diagnosis"]

    sp_pt <- SpatialPlot(comb_seurat, group.by="layer_label_v2", image.alpha=0.3, images=img, stroke=NA) + scale_fill_manual(values=group.colors) + ggtitle(paste(sample_id, diagnosis))
    print(sp_pt)
}
dev.off()

saveRDS(comb_seurat, "comb_seurat_diet_out_plusL4_V4_seed101_11092023.rds")


generate_layer_corr_table <- function(top_markers, maynard, maynard_t_stats) {
    corr_table <- as.data.frame(matrix(ncol=length(unique(top_markers$cluster)),
                                       nrow=length(maynard_t_stats)))
    rownames(corr_table)<-maynard_t_stats
    colnames(corr_table)<-unique(top_markers$cluster)

    corr_table_pvals <- as.data.frame(matrix(ncol=length(unique(top_markers$cluster)),
                                             nrow=length(maynard_t_stats)))
    rownames(corr_table_pvals)<-maynard_t_stats
    colnames(corr_table_pvals)<-unique(top_markers$cluster)

    for (i in unique(top_markers$cluster)) {
        sig_genes_layer <- subset(top_markers, cluster==i)
        # sig_genes_layer$ranked_LFC <- rank(sig_genes_layer$avg_log2FC)
        for(j in maynard_t_stats) {
            # maynard_ranked_col <- paste("ranked_", j, sep='')
            # merged_mat <- merge(sig_genes_layer, maynard[, c(maynard_ranked_col, "gene")], by="gene")
            merged_mat <- merge(sig_genes_layer, maynard[, c(j, "gene")], by="gene")
            # corr_table[j,i] <- cor(merged_mat$ranked_LFC, merged_mat[[maynard_ranked_col]])
            # res <- cor.test(merged_mat$ranked_LFC, merged_mat[[maynard_ranked_col]])
            # res <- cor.test(merged_mat$avg_log2FC, merged_mat[[j]], method="spearman")
            res <- cor.test(merged_mat$avg_log2FC, merged_mat[[j]], method="pearson")
            corr_table[j,i] <- res$estimate
            corr_table_pvals[j,i] <- res$p.value
        }
    }
    results <- list()
    results$pearson <- corr_table
    results$pvalues <- corr_table_pvals
    return(results)
}

permute_k_subset <- function(set, k) {
    permut_mat <- c()
    permute_k_helper <- function(chain, remaining, k) {
        if ( length(chain) == k | length(remaining) == 0 ) {
            # print(chain)
            permut_mat <<- rbind(permut_mat, chain)
        } else {
            for (i in 1:length(remaining)) {
                permute_k_helper(c(chain, remaining[i]), remaining[-i], k)
            }
        }
    }
    permute_k_helper(c(), set, k)
    rownames(permut_mat) <- NULL
    return(permut_mat)
}



maynard <- read.csv("/data/neurogen/jy1008/scRNAseq_integration/maynard_layer_markers.csv", sep=',', header=TRUE)
maynard_t_stats <- c("t_stat_WM", "t_stat_Layer1", "t_stat_Layer2", "t_stat_Layer3",
                     "t_stat_Layer4", "t_stat_Layer5", "t_stat_Layer6")

set.seed(101)

# don't use every gene for correlation
Idents(comb_seurat_all) <- comb_seurat_all@meta.data[, "new_label"]
top_markers <- RunPrestoAll(comb_seurat_all, assay='Spatial', slot='data', logfc.threshold=0.1, return.thresh=0.1, min.pct = 0.1)

write.table(top_markers, "top_markers_newlayer4_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)

corr_table <- generate_layer_corr_table(top_markers, maynard, maynard_t_stats)
write.table(corr_table$pearson, "maynard_validation_pearson_newlayer4_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)
write.table(corr_table$pvalues, "maynard_validation_pvalues_newlayer4_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)




# compare to the old labels (w/out Layer 4)
Idents(comb_seurat_all) <- comb_seurat_all@meta.data[, "layer_label"]
top_markers <- RunPrestoAll(comb_seurat_all, assay='Spatial', slot='data', logfc.threshold=0.1, return.thresh=0.1, min.pct = 0.1)

write.table(top_markers, "top_markers_oldlabels_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)

corr_table <- generate_layer_corr_table(top_markers, maynard, maynard_t_stats)
write.table(corr_table$pearson, "maynard_validation_pearson_oldlabels_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)
write.table(corr_table$pvalues, "maynard_validation_pvalues_oldlabels_all_data_5covars0p4_seed101_res0p45_12142022.csv", sep = ',', row.names = TRUE, col.names = NA)










validate_by_resolution <- function(seurat_obj, maynard_markers, output_name, small_cluster_thresh = 100) {

    # validation against Maynard for each Leiden cluster resolution
    # maynard <- read.csv("/data/neurogen/jy1008/scRNAseq_integration/maynard_layer_markers.csv", sep=',', header=TRUE)

    maynard_t_stats <- c("t_stat_WM", "t_stat_Layer1", "t_stat_Layer2", "t_stat_Layer3",
                         "t_stat_Layer4", "t_stat_Layer5", "t_stat_Layer6")

    clust_cols <- colnames(seurat_obj@meta.data)
    clust_cols <- clust_cols[startsWith(clust_cols, "Spatial_snn_res")]
    
    cluster_res_table <- NULL
    for( clust_col in clust_cols ) {
        print(clust_col)
        # fold small clusters into a single "noise" cluster
        seurat_obj@meta.data[, "temp_filt_cluster"] <- "Noise"
        for (val in unique(seurat_obj@meta.data[, clust_col])) {
            val_count <- sum(seurat_obj@meta.data[, clust_col] == val)
            if (val_count > small_cluster_thresh) {
                seurat_obj@meta.data[seurat_obj@meta.data[, clust_col] == val,
                                           "temp_filt_cluster"] <- val
            }
        }
        # skip if fewer than 5 clusters
        num_clusters <- length(unique(seurat_obj@meta.data[, "temp_filt_cluster"]))
        if ( num_clusters < 5 ) {
            print("less than 5 clusters produced")
            continue
        }
        # top_markers <- RunPrestoAll(seurat_obj, assay='Spatial', slot='data', group.by="temp_filt_cluster", logfc.threshold=0, return.thresh=1.0, min.pct = 0.0, min.cells.feature = 0, min.cells.group = 0)
        
        # don't use every gene for correlation
        temp_table <- table(seurat_obj@meta.data[, "temp_filt_cluster"])
        print(temp_table)
        Idents(seurat_obj) <- seurat_obj@meta.data[, "temp_filt_cluster"]
        top_markers <- RunPrestoAll(seurat_obj, assay='Spatial', slot='data', logfc.threshold=0.1, return.thresh=0.1, min.pct = 0.1)
        
        write.table(top_markers, sprintf("top_markers_%s_%s.csv", output_name, clust_col), sep = ',', row.names = TRUE, col.names = NA)
        
        corr_table <- generate_layer_corr_table(top_markers, maynard, maynard_t_stats)
        write.table(corr_table$pearson, sprintf("maynard_validation_pearson_%s_%s.csv", output_name, clust_col), sep = ',', row.names = TRUE, col.names = NA)
        write.table(corr_table$pvalues, sprintf("maynard_validation_pvalues_%s_%s.csv", output_name, clust_col), sep = ',', row.names = TRUE, col.names = NA)
        
        # find the best layer-cluster
        # clusters to validate layers: 1, 2, 3 (2/3?), (4?), 5, 6
        cluster_orders <- permute_k_subset(colnames(corr_table$pearson), 5)
        best_cluster_order <- NULL
        best_corr_sum <- -999
        for (r in 1:nrow(cluster_orders)) {
            corr_sum <- corr_table$pearson["t_stat_Layer1", cluster_orders[r, 1]] +
                        corr_table$pearson["t_stat_Layer2", cluster_orders[r, 2]] +
                        corr_table$pearson["t_stat_Layer3", cluster_orders[r, 3]] +
                        corr_table$pearson["t_stat_Layer5", cluster_orders[r, 4]] +
                        corr_table$pearson["t_stat_Layer6", cluster_orders[r, 5]]
            print(corr_sum)
            if (corr_sum > best_corr_sum) {
                best_corr_sum <- corr_sum
                best_cluster_order <- cluster_orders[r, ]
            }
        }
        print(best_cluster_order)
        print(best_corr_sum)
        
        # within-cluster distance sum for assigned layer clusts
        # ave_expr <- AverageExpression(seurat_obj, assays = "Spatial",
        #                               features = VariableFeatures(seurat_obj),
        #                               group.by = clust_col, slot = "data")$Spatial
        layer_dist_sum <- 0
        for ( layer_clust in best_cluster_order ) {
            expr_mat <- GetAssayData(seurat_obj, assay = "Spatial", slot = "data")[
                                            VariableFeatures(seurat_obj),
                                            seurat_obj@meta.data[, "temp_filt_cluster"] == layer_clust]
            expr_mat <- sweep(expr_mat, 1, rowMeans(expr_mat), "-")
            layer_dist_sum <- layer_dist_sum + sum(colSums(expr_mat^2))
        }
        print(layer_dist_sum)
        
        cluster_res_table <- rbind(cluster_res_table,
                                   data.frame(res = clust_col,
                                              best_cluster_order = paste(best_cluster_order, collapse=','),
                                              best_maynard_corr_sum = best_corr_sum,
                                              layer_sum_squared_distance = layer_dist_sum,
                                              num_large_clusters = num_clusters))
    }
    # write.table(cluster_res_table, "cluster_res_table_all_conts_07012022.csv", sep=',')
    write.table(cluster_res_table, sprintf("cluster_res_table_%s.csv", output_name), sep = ',', row.names = FALSE)
}










# comb_seurat_cont <- readRDS("/data/neurogen/jy1008/scRNAseq_integration/spatial_clustering_pipeline_06252022/all_controls_5covars0p4_seed101_05082022/comb_seurat_diet_out.rds")
# save some memory
# comb_seurat_cont <- DietSeurat(comb_seurat_cont)

comb_seurat <- readRDS("/data/neurogen/jy1008/scRNAseq_integration/spatial_clustering_pipeline_06252022/all_data_5covars0p4_seed101_06252022/comb_seurat_diet_out_no_reducs.rds")

maynard <- read.csv("/data/neurogen/jy1008/scRNAseq_integration/maynard_layer_markers.csv", sep=',', header=TRUE)

# out_name <- "all_conts_12122022"
out_name <- "all_data_12122022"

set.seed(101)
validate_by_resolution(comb_seurat, maynard_markers = maynard,
                       output_name = out_name,
                       small_cluster_thresh = 100)






# ###
# # NOTE: TEMPORARY
# # Recalculate the cluster distances as sum of squared distances
# ###
# # -----------------------------------------------------------------------------
# seurat_obj <- comb_seurat_cont
# small_cluster_thresh <- 100
# for (i in 1:nrow(cluster_res_table)) {
#     clust_col <- cluster_res_table[i, "res"]
#     print(clust_col)
#
#     seurat_obj@meta.data[, "temp_filt_cluster"] <- "Noise"
#     for (val in unique(seurat_obj@meta.data[, clust_col])) {
#         val_count <- sum(seurat_obj@meta.data[, clust_col] == val)
#         if (val_count > small_cluster_thresh) {
#             seurat_obj@meta.data[seurat_obj@meta.data[, clust_col] == val,
#                                        "temp_filt_cluster"] <- val
#         }
#     }
#     # skip if fewer than 5 clusters
#     num_clusters <- length(unique(seurat_obj@meta.data[, "temp_filt_cluster"]))
#     if ( num_clusters < 5 ) {
#         print("less than 5 clusters produced")
#         continue
#     }
#     best_clust_ord <- unlist(strsplit(cluster_res_table[i, "best_cluster_order"], ','))
#     layer_dist_sum <- 0
#     for ( layer_clust in best_clust_ord ) {
#         expr_mat <- GetAssayData(seurat_obj, assay = "Spatial", slot = "data")[
#                                         VariableFeatures(seurat_obj),
#                                         seurat_obj@meta.data[, "temp_filt_cluster"] == layer_clust]
#         expr_mat <- sweep(expr_mat, 1, rowMeans(expr_mat), "-")
#         print(expr_mat[1:5,1:5])
#         layer_dist_sum <- layer_dist_sum + sum(colSums(expr_mat^2))
#     }
#     print(layer_dist_sum)
#     cluster_res_table[i, "layer_sum_squared_distance"] <- layer_dist_sum
# }
# cluster_res_table$layer_distance_sum <- NULL
# cluster_res_table <- cluster_res_table[, c("res", "best_cluster_order",
#                         "best_maynard_corr_sum", "layer_sum_squared_distance",
#                         "num_large_clusters")]
# # -----------------------------------------------------------------------------

# cluster_res_table <- read.csv("cluster_res_table_all_conts_07012022.csv")
# cluster_res_table <- read.csv("cluster_res_table_all_conts_12122022.csv")
cluster_res_table <- read.csv("cluster_res_table_all_data_12122022.csv")
cluster_res_table$cluster_resolution <- as.numeric(gsub("Spatial_snn_res\\.", "", cluster_res_table$res))


# https://waterdata.usgs.gov/blog/beyond-basic-plotting/
# pdf("cluster_res_validate_plot_all_conts_12122022.pdf")
pdf("cluster_res_validate_plot_all_data_12122022.pdf")
temp <- melt(cluster_res_table, measure.vars = c("best_maynard_corr_sum", "layer_sum_squared_distance", "num_large_clusters"))
pt1 <- ggplot(temp, aes(x = cluster_resolution, y = value)) +
                geom_line(aes(color = variable))
pt2 <- pt1 + facet_grid(variable ~ ., scales = "free",
switch = "y", labeller = as_labeller(c(num_large_clusters = "Number of large clusters", layer_sum_squared_distance = "Sum sq dist from centroids", best_maynard_corr_sum = "Validation sum of correlations"))) + theme_bw() + ylab(NULL) +
theme(strip.background = element_blank(), strip.placement = "outside",
      legend.position = "none",
axis.line = element_line(colour = "black")) + scale_x_continuous(limits=c(min(cluster_res_table$cluster_resolution), max(cluster_res_table$cluster_resolution)))
print(pt2)
dev.off()

