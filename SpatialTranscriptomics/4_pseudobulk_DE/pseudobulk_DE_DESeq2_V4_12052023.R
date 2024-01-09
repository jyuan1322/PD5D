library(Seurat)
library(DESeq2)
library(stringr)
library(testit)

# layer-specific DE
# -- controls
# -- all samples
# -- smoothed or unsmoothed labels
# -- one layer vs all others

# layer-wise clinical DE
# -- nctx_temporal (principal phenotype)
# -- brain_stem_sn (early PD)
# -- last MMSE score (mental)
# -- motor UPDRS (motor)
# -- control/ILBD/case


# [1] "orig.ident"           "nCount_Spatial"       "nFeature_Spatial"
# [4] "sample_name"          "batch"                "PMI"
# [7] "RIN"                  "sex"                  "expired_age"
# [10] "diagnosis"            "last_mmse_test_score" "motor_updrs_score"
# [13] "nctx_temporal"        "brain_stem_sn"        "selected_spot"
# [16] "PMI_num"              "RIN_num"              "expired_age_num"
# [19] "Spatial_snn_res.0.3"  "seurat_clusters"      "Spatial_snn_res.0.35"
# [22] "Spatial_snn_res.0.4"  "Spatial_snn_res.0.45" "Spatial_snn_res.0.5"
# [25] "Spatial_snn_res.0.55" "Spatial_snn_res.0.6"  "layer_label_v1"
# [28] "layer_label_v2"       "smoothed_label_s1"    "s2_subclust"
# [31] "s2_subclust_score"    "s2_nbr_score"         "smoothed_label_s2"
# [34] "smoothed_label_s3"    "smoothed_label_s4"    "smoothed_label_s5"

comb_seurat <- readRDS("/mnt/data0/projects/ASAP/ST/clustering_pipeline/layer_smoothing_V4_seed101_11302023_replication_test/comb_seurat_post_smoothed_label_s5_11302023.rds")

random_seed <- 101

# Map diagnosis to pseudotime: Control = 0, ILBD = 1, Case = 2
comb_seurat@meta.data$diagnosis.int <- NA
comb_seurat@meta.data$diagnosis.int[comb_seurat@meta.data$diagnosis == "Control"] <- 0
comb_seurat@meta.data$diagnosis.int[comb_seurat@meta.data$diagnosis == "ILBD"] <- 1
comb_seurat@meta.data$diagnosis.int[comb_seurat@meta.data$diagnosis == "Case"] <- 2
comb_seurat@meta.data$diagnosis.int <- as.numeric(comb_seurat@meta.data$diagnosis.int)

# decide which layer annotation to use: smoothed or unsmoothed
layer_annot <- "smoothed_label_s5"
layers <- sort(unique(comb_seurat@meta.data[, layer_annot]))
features_quantitative <- c("nctx_temporal", "brain_stem_sn", "last_mmse_test_score", "motor_updrs_score")
min_layer_spot_count <- 20

# loop over layers: create seurat object for each
for(layer in layers) {
    temp_layer <- FetchData(comb_seurat, vars = layer_annot)
    comb_sub <- comb_seurat[, which(temp_layer == layer)]
    Idents(object = comb_sub) <- "sample_name"
    
    # get countData and filter out lowly expressed genes
    countData <- AggregateExpression(comb_sub, assays="Spatial", slot="counts")
    countData <- countData$Spatial
    keep_genes <- rowSums(countData >= 5) >= 3
    countData <- countData[keep_genes, ]
    
    # filter out samples with fewer than N spots belonging to the given layer
    keep_samples <- names( which( table(comb_sub@meta.data$sample_name) > min_layer_spot_count ) )
    countData <- countData[, keep_samples]

    count_out_csv <- sprintf("agg_counts_filtered_%s_%s_V4_12052023.csv", layer_annot, gsub(' ', '', layer))
    if(!file.exists(count_out_csv)) {
        write.csv(countData, file = count_out_csv)
    }
    
    # for each feature, create metadata table
    for(feat_quant in features_quantitative) {
        print(paste("Running", layer, feat_quant))
        
        sample_meta <- comb_sub@meta.data[, c("sample_name", "batch", "sex", "expired_age_num", "PMI_num", "RIN_num", feat_quant)]
        sample_meta <- sample_meta[!duplicated(sample_meta$sample_name), ]
        sample_meta <- sample_meta[sample_meta$sample_name %in% colnames(countData), ]
        # filter out samples whose feature label is NA
        sample_meta <- sample_meta[!is.na(sample_meta[, feat_quant]), ]
        rownames(sample_meta) <- sample_meta$sample_name
        countData_filt <- countData[, as.character(sample_meta$sample_name)]
        print(paste("Samples passing filtering", ncol(countData_filt)))
        
        assert(all(rownames(sample_meta) == colnames(countData_filt)))
        
        # DESeq2 likes to have all numeric variables centered and scaled
        sample_meta$sample_name <- as.factor(sample_meta$sample_name)
        sample_meta$batch <- as.factor(sample_meta$batch)
        sample_meta$sex <- as.factor(sample_meta$sex)
        sample_meta$PMI_num <- scale(sample_meta$PMI_num, center=TRUE, scale=TRUE)
        sample_meta$RIN_num <- scale(sample_meta$RIN_num, center=TRUE, scale=TRUE)
        sample_meta$expired_age_num <- scale(sample_meta$expired_age_num, center=TRUE, scale=TRUE)
        sample_meta[, feat_quant] <- scale(sample_meta[, feat_quant], center=TRUE, scale=TRUE)
        
        assert(is.factor(sample_meta$sample_name))
        assert(is.factor(sample_meta$batch))
        assert(is.factor(sample_meta$sex))
        assert(is.numeric(sample_meta$expired_age_num))
        assert(is.numeric(sample_meta$PMI_num))
        assert(is.numeric(sample_meta$RIN_num))
        assert(is.numeric(sample_meta[, feat_quant]))
        
        
        tryCatch(
            expr = {

                set.seed(random_seed)
                dds <- DESeqDataSetFromMatrix(countData=countData_filt, colData=sample_meta, design= model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num + sample_meta[, feat_quant]), tidy=FALSE)
                saveRDS(dds, sprintf("pseudobulk_clinical_deseq2_dds_%s_%s_12052023.rds", feat_quant, gsub(' ', '', layer)))

                dds_lrt <- DESeq(dds, test="LRT", reduced = model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num))
                saveRDS(dds_lrt, sprintf("pseudobulk_clinical_deseq2_dds_lrt_%s_%s_12052023.rds", feat_quant, gsub(' ', '', layer)))

                results_table <- results(dds_lrt)

                write.csv(results_table, sprintf("pseudobulk_clinical_deseq2_%s_%s_12052023.csv", feat_quant, gsub(' ', '', layer)))

            },
            error = function(e){
                # (Optional)
                # Do this if an error is caught...
                print(paste("Running", layer, feat_quant))
                message(e)
            }
        )

    }
}

# [1] "orig.ident"           "nCount_Spatial"       "nFeature_Spatial"
# [4] "sample_name"          "batch"                "PMI"
# [7] "RIN"                  "sex"                  "expired_age"
# [10] "diagnosis"            "last_mmse_test_score" "motor_updrs_score"
# [13] "nctx_temporal"        "brain_stem_sn"        "selected_spot"
# [16] "PMI_num"              "RIN_num"              "expired_age_num"
# [19] "Spatial_snn_res.0.3"  "seurat_clusters"      "Spatial_snn_res.0.35"
# [22] "Spatial_snn_res.0.4"  "Spatial_snn_res.0.45" "Spatial_snn_res.0.5"
# [25] "Spatial_snn_res.0.55" "Spatial_snn_res.0.6"  "layer_label_v1"
# [28] "layer_label_v2"       "smoothed_label_s1"    "s2_subclust"
# [31] "s2_subclust_score"    "s2_nbr_score"         "smoothed_label_s2"
# [34] "smoothed_label_s3"    "smoothed_label_s4"    "smoothed_label_s5"





# Control/ILBD and Control/Case comparisons

comb_seurat <- readRDS("/mnt/data0/projects/ASAP/ST/clustering_pipeline/layer_smoothing_V4_seed101_11302023_replication_test/comb_seurat_post_smoothed_label_s5_11302023.rds")

# filter bad samples
# bad_samples <- c("BN0329", "BN0602_rep1", "BN1750")
# bad_samples <- c("BN0329", "BN0602_rep1", "BN1750", "BN1204", "BN1412_rep1", "BN0635", "BN1412_rep3", "BN1959", "BN9950", "BN1022_rep2", "BN1412_rep2", "BN1554_rep1", "BN1867", "BN1317", "BN0602_rep2")
# comb_seurat_original <- comb_seurat
# comb_suerat <- comb_seurat[, which(!(comb_seurat@meta.data$orig.ident %in% bad_samples))]

random_seed <- 101

# in nctx_temporal, there is only 1 sample with a score of 4: set this to 3 so as not to overly bias the regression
comb_seurat@meta.data$nctx_temporal_fix <- comb_seurat@meta.data$nctx_temporal
comb_seurat@meta.data$nctx_temporal_fix[comb_seurat@meta.data$nctx_temporal_fix > 3] <- 3

# decide which layer annotation to use: smoothed or unsmoothed
layer_annot <- "smoothed_label_s5"
layers <- sort(unique(comb_seurat@meta.data[, layer_annot]))
# features_quantitative <- c("nctx_temporal_fix", "brain_stem_sn", "last_mmse_test_score", "motor_updrs_score")
min_layer_spot_count <- 20

comb_seurat_original <- comb_seurat
# control vs ILBD
feat_diag <- "diagnosis_cont_ilbd"
comb_seurat <- comb_seurat[, which(comb_seurat@meta.data$diagnosis %in% c("Control", "ILBD"))]
comb_seurat@meta.data[, feat_diag] <- factor(comb_seurat@meta.data$diagnosis, levels=c("Control", "ILBD"))


# debugging matrix rank error (batch?)
temp <- comb_seurat@meta.data[, c("orig.ident", "batch", "diagnosis_cont_ilbd")]
temp <- temp[match(unique(temp$orig.ident), temp$orig.ident),]
table(temp[, c("batch", "diagnosis_cont_ilbd")])
#      diagnosis_cont_ilbd
# batch Control ILBD
#    1        1    0
#    10       1    1
#    11       1    1
#    12       1    1
#    13       1    1
#    14       1    2
#    15       1    2
#    16       1    2
#    18       1    2
#    19       1    3
#    2        2    0
#    20       1    2
#    21       2    2
#    22       1    1
#    23       1    1
#    24       2    0
#    25       2    0
#    26       2    1
#    27       0    0
#    3        1    1
#    4        2    0
#    5        1    2
#    6        2    1
#    7        1    1
#    8        1    2
#    81       0    0
#    9        1    1

# loop over layers: create seurat object for each
for(layer in layers) {
    temp_layer <- FetchData(comb_seurat, vars = layer_annot)
    comb_sub <- comb_seurat[, which(temp_layer == layer)]
    Idents(object = comb_sub) <- "sample_name"
    
    # get countData and filter out lowly expressed genes
    countData <- AggregateExpression(comb_sub, assays="Spatial", slot="counts")
    countData <- countData$Spatial
    keep_genes <- rowSums(countData >= 5) >= 3
    countData <- countData[keep_genes, ]
    
    # filter out samples with fewer than N spots belonging to the given layer
    keep_samples <- names( which( table(comb_sub@meta.data$sample_name) > min_layer_spot_count ) )
    countData <- countData[, keep_samples]

    # count_out_csv <- sprintf("agg_counts_filtered_%s_%s_V4_12052023.csv", layer_annot, gsub(' ', '', layer))
    # if(!file.exists(count_out_csv)) {
    #     write.csv(countData, file = count_out_csv)
    # }
    
    # for each feature, create metadata table
    print(paste("Running", layer, feat_diag))
    
    sample_meta <- comb_sub@meta.data[, c("sample_name", "batch", "sex", "expired_age_num", "PMI_num", "RIN_num", feat_diag)]
    sample_meta <- sample_meta[!duplicated(sample_meta$sample_name), ]
    sample_meta <- sample_meta[sample_meta$sample_name %in% colnames(countData), ]
    # filter out samples whose feature label is NA
    sample_meta <- sample_meta[!is.na(sample_meta[, feat_diag]), ]
    rownames(sample_meta) <- sample_meta$sample_name
    countData_filt <- countData[, as.character(sample_meta$sample_name)]
    print(paste("Samples passing filtering", ncol(countData_filt)))
    
    assert(all(rownames(sample_meta) == colnames(countData_filt)))
    
    # DESeq2 likes to have all numeric variables centered and scaled
    sample_meta$sample_name <- as.factor(sample_meta$sample_name)
    sample_meta$batch <- as.factor(sample_meta$batch)
    sample_meta$sex <- as.factor(sample_meta$sex)
    sample_meta$PMI_num <- scale(sample_meta$PMI_num, center=TRUE, scale=TRUE)
    sample_meta$RIN_num <- scale(sample_meta$RIN_num, center=TRUE, scale=TRUE)
    sample_meta$expired_age_num <- scale(sample_meta$expired_age_num, center=TRUE, scale=TRUE)
    # sample_meta[, feat_quant] <- scale(sample_meta[, feat_quant], center=TRUE, scale=TRUE)
    
    assert(is.factor(sample_meta$sample_name))
    assert(is.factor(sample_meta$batch))
    assert(is.factor(sample_meta$sex))
    assert(is.numeric(sample_meta$expired_age_num))
    assert(is.numeric(sample_meta$PMI_num))
    assert(is.numeric(sample_meta$RIN_num))
    # assert(is.numeric(sample_meta[, feat_quant]))
    assert(is.factor(sample_meta[, feat_diag]))
    
    tryCatch(
        expr = {

            set.seed(random_seed)
            dds <- DESeqDataSetFromMatrix(countData=countData_filt, colData=sample_meta, design= model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num + sample_meta[, feat_diag]), tidy=FALSE)
            saveRDS(dds, sprintf("pseudobulk_clinical_deseq2_dds_%s_%s_12052023v4.rds", feat_diag, gsub(' ', '', layer)))

            # SVA?
            dds <- DESeqDataSetFromMatrix(countData=countData_filt, colData=sample_meta, design= model.matrix(~ sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num + sample_meta[, feat_diag]), tidy=FALSE)
            dds <- estimateSizeFactors(dds)
            dat  <- counts(dds, normalized = TRUE)
            idx  <- rowMeans(dat) > 1
            dat  <- dat[idx, ]
            mod  <- model.matrix(~ PMI_num + RIN_num + sex + expired_age_num + diagnosis_cont_ilbd, colData(dds))
            mod0 <- model.matrix(~ PMI_num + RIN_num + sex + expired_age_num, colData(dds))

            
            # The number of SVs can be the number of unique batches
            # https://support.bioconductor.org/p/71447/#87007
            n.sv <- num.sv(dat,mod,method="leek") # 55
            n.sv <- num.sv(dat,mod,method="be") # 1 why so different?
            svseq <- svaseq(dat, mod, mod0, n.sv = 25)
            
            ddssva <- dds
            ddssva$SV1 <- svseq$sv[,1]
            ddssva$SV2 <- svseq$sv[,2]
            design(ddssva) <- ~ SV1 + SV2 + dex
            
            
            
            dds_lrt <- DESeq(dds, test="LRT", reduced = model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num))
            saveRDS(dds_lrt, sprintf("pseudobulk_clinical_deseq2_dds_lrt_%s_%s_12052023v4.rds", feat_diag, gsub(' ', '', layer)))

            results_table <- results(dds_lrt)

            write.csv(results_table, sprintf("pseudobulk_clinical_deseq2_%s_%s_12052023v4.csv", feat_diag, gsub(' ', '', layer)))

        },
        error = function(e){
            # (Optional)
            # Do this if an error is caught...
            print(paste("Running", layer, feat_diag))
            message(e)
        }
    )

}




# diagnosis as ordered factor
comb_seurat <- readRDS("/mnt/data0/projects/ASAP/ST/clustering_pipeline/layer_smoothing_V4_seed101_11302023_replication_test/comb_seurat_post_smoothed_label_s5_11302023.rds")

random_seed <- 101

# map diagnosis to ordered factor
comb_seurat@meta.data$diagnosis <- factor(comb_seurat@meta.data$diagnosis, ordered=TRUE, levels=c("Control", "ILBD", "Case"))

# decide which layer annotation to use: smoothed or unsmoothed
layer_annot <- "smoothed_label_s5"
layers <- sort(unique(comb_seurat@meta.data[, layer_annot]))
features_ordered <- c("diagnosis")
min_layer_spot_count <- 20

# loop over layers: create seurat object for each
for(layer in layers) {
    temp_layer <- FetchData(comb_seurat, vars = layer_annot)
    comb_sub <- comb_seurat[, which(temp_layer == layer)]
    Idents(object = comb_sub) <- "sample_name"
    
    # get countData and filter out lowly expressed genes
    countData <- AggregateExpression(comb_sub, assays="Spatial", slot="counts")
    countData <- countData$Spatial
    keep_genes <- rowSums(countData >= 5) >= 3
    countData <- countData[keep_genes, ]
    
    # filter out samples with fewer than N spots belonging to the given layer
    keep_samples <- names( which( table(comb_sub@meta.data$sample_name) > min_layer_spot_count ) )
    countData <- countData[, keep_samples]

    count_out_csv <- sprintf("agg_counts_filtered_%s_%s_V4_12052023.csv", layer_annot, gsub(' ', '', layer))
    if(!file.exists(count_out_csv)) {
        write.csv(countData, file = count_out_csv)
    }
    
    # for each feature, create metadata table
    for(feat_ordered in features_ordered) {
        print(paste("Running", layer, feat_ordered))
        
        sample_meta <- comb_sub@meta.data[, c("sample_name", "batch", "sex", "expired_age_num", "PMI_num", "RIN_num", feat_ordered)]
        sample_meta <- sample_meta[!duplicated(sample_meta$sample_name), ]
        sample_meta <- sample_meta[sample_meta$sample_name %in% colnames(countData), ]
        # filter out samples whose feature label is NA
        sample_meta <- sample_meta[!is.na(sample_meta[, feat_ordered]), ]
        rownames(sample_meta) <- sample_meta$sample_name
        countData_filt <- countData[, as.character(sample_meta$sample_name)]
        print(paste("Samples passing filtering", ncol(countData_filt)))
        
        assert(all(rownames(sample_meta) == colnames(countData_filt)))
        
        # DESeq2 likes to have all numeric variables centered and scaled
        sample_meta$sample_name <- as.factor(sample_meta$sample_name)
        sample_meta$batch <- as.factor(sample_meta$batch)
        sample_meta$sex <- as.factor(sample_meta$sex)
        sample_meta$PMI_num <- scale(sample_meta$PMI_num, center=TRUE, scale=TRUE)
        sample_meta$RIN_num <- scale(sample_meta$RIN_num, center=TRUE, scale=TRUE)
        sample_meta$expired_age_num <- scale(sample_meta$expired_age_num, center=TRUE, scale=TRUE)
        # sample_meta[, feat_quant] <- scale(sample_meta[, feat_quant], center=TRUE, scale=TRUE)
        
        
        assert(is.factor(sample_meta$sample_name))
        assert(is.factor(sample_meta$batch))
        assert(is.factor(sample_meta$sex))
        assert(is.numeric(sample_meta$expired_age_num))
        assert(is.numeric(sample_meta$PMI_num))
        assert(is.numeric(sample_meta$RIN_num))
        # assert(is.numeric(sample_meta[, feat_quant]))
        assert(is.factor(sample_meta[, feat_ordered]))
        assert(is.ordered(sample_meta[, feat_ordered]))
        
        tryCatch(
            expr = {

                set.seed(random_seed)
                dds <- DESeqDataSetFromMatrix(countData=countData_filt, colData=sample_meta, design= model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num + sample_meta[, feat_ordered]), tidy=FALSE)
                saveRDS(dds, sprintf("pseudobulk_clinical_deseq2_dds_%s_%s_12122023.rds", feat_ordered, gsub(' ', '', layer)))

                dds_lrt <- DESeq(dds, test="LRT", reduced = model.matrix(~ sample_meta$batch + sample_meta$PMI_num + sample_meta$RIN_num + sample_meta$sex + sample_meta$expired_age_num))
                saveRDS(dds_lrt, sprintf("pseudobulk_clinical_deseq2_dds_lrt_%s_%s_12122023.rds", feat_ordered, gsub(' ', '', layer)))

                results_table <- results(dds_lrt)

                write.csv(results_table, sprintf("pseudobulk_clinical_deseq2_%s_%s_12122023.csv", feat_ordered, gsub(' ', '', layer)))

            },
            error = function(e){
                # (Optional)
                # Do this if an error is caught...
                print(paste("Running", layer, feat_ordered))
                message(e)
            }
        )

    }
}
















# plot gene vs trait
feat_quant <- "nctx_temporal"
layer_annot <- "smoothed_label_s5"
layer <- "Layer 6"

temp_layer <- FetchData(comb_seurat, vars = layer_annot)
comb_sub <- comb_seurat[, which(temp_layer == layer)]
Idents(object = comb_sub) <- "sample_name"

# get countData and filter out lowly expressed genes
# countData <- AggregateExpression(comb_sub, assays="Spatial", slot="data")
countData <- AverageExpression(comb_sub, assays="Spatial", slot="data")
countData <- countData$Spatial
# keep_genes <- rowSums(countData >= 5) >= 3
# countData <- countData[keep_genes, ]

# filter out samples with fewer than N spots belonging to the given layer
keep_samples <- names( which( table(comb_sub@meta.data$sample_name) > min_layer_spot_count ) )
countData <- countData[, keep_samples]

sample_meta <- comb_sub@meta.data[, c("sample_name", "batch", "sex", "expired_age_num", "PMI_num", "RIN_num", feat_quant)]
sample_meta <- sample_meta[!duplicated(sample_meta$sample_name), ]
sample_meta <- sample_meta[sample_meta$sample_name %in% colnames(countData), ]
# filter out samples whose feature label is NA
sample_meta <- sample_meta[!is.na(sample_meta[, feat_quant]), ]
rownames(sample_meta) <- sample_meta$sample_name
countData_filt <- countData[, as.character(sample_meta$sample_name)]
print(paste("Samples passing filtering", ncol(countData_filt)))

assert(all(rownames(sample_meta) == colnames(countData_filt)))


pdf("snca_debug.pdf")
sig_gene <- "SNCA"
temp_table <- cbind(sample_meta, sig_gene = countData_filt[sig_gene, ])

# pt <- ggplot(temp_table, aes(x=nctx_temporal, y=sig_gene)) + geom_boxplot() + theme_classic() + ylab(sig_gene)
pt <- ggplot(temp_table, aes(x=as.factor(nctx_temporal), y=sig_gene)) + geom_boxplot() + theme_classic() + ylab(sig_gene)
print(pt)
dev.off()


























score_layer_fit <- function(comb_seurat, cluster_label_name, spot_label_to_eval=NULL, image_name=NULL, verbose=FALSE) {
    allowed_neighbors <- list(
        "Layer 1" = c("Layer 1", "Layer 2"),
        "Layer 2" = c("Layer 1", "Layer 2", "Layer 3"),
        "Layer 3" = c("Layer 2", "Layer 3", "Layer 4"),
        "Layer 4" = c("Layer 3", "Layer 4", "Layer 5"),
        "Layer 5" =  c("Layer 4", "Layer 5", "Layer 6"),
        "Layer 6" = c("Layer 5", "Layer 6", "WM"),
        "WM" = c("Layer 6", "WM"))

    cluster_labels <- comb_seurat@meta.data[, c("orig.ident", cluster_label_name)]

    layer_scores <- c()
    if(is.null(image_name)) {
        img_list <- names(comb_seurat@images)
    } else {
        img_list <- c(image_name)
    }
    for (img in img_list) {
        
        meta_row <- rownames(comb_seurat@images[[img]]@coordinates)[1]
        sample_name_filt <- as.character(comb_seurat@meta.data[meta_row, "orig.ident"])
        
        coord_table <- comb_seurat@images[[img]]@coordinates
        coord_table <- merge(coord_table, cluster_labels, by=0)
        rownames(coord_table) <- coord_table$Row.names
        
        # hexagonal
        coord_table$euclid_x <- (coord_table$row %% 2) / 2 + (coord_table$col - (coord_table$col %% 2)) / 2
        coord_table$euclid_y <- sqrt(3) * ( (coord_table$row %% 2) / 2 + (coord_table$row - (coord_table$row %% 2)) / 2)

        coord_dists <- dist(coord_table[, c("euclid_x", "euclid_y")])
        coord_dists <- as.matrix(coord_dists, nrow=nrow(coord_table))
        # round to nearest 0.1
        coord_dists <- round(coord_dists * 10) / 10
        good_adjs <- 0
        total_adjs <- 0
        for(spot_id in rownames(coord_table)) {
            spot_label <- coord_table[spot_id, cluster_label_name]
            if(!is.null(spot_label_to_eval) && spot_label == spot_label_to_eval) {
                nbr_ids <- colnames(coord_dists)[ which( abs(coord_dists[spot_id, ] - 1) < 0.01 ) ]
                nbr_labels <- coord_table[nbr_ids, cluster_label_name]
                good_adjs <- good_adjs + sum(nbr_labels %in% allowed_neighbors[[spot_label]])
                total_adjs <- total_adjs + length(nbr_labels)
                if(verbose) {
                    print('-----')
                    print(paste(spot_id, spot_label))
                    print(nbr_ids)
                    print(nbr_labels)
                    print(sum(nbr_labels %in% allowed_neighbors[[spot_label]]))
                }
            }
        }
        layer_scores <- rbind(layer_scores, c(sample_name_filt, (good_adjs / total_adjs)))
    }
    return(layer_scores)
}
