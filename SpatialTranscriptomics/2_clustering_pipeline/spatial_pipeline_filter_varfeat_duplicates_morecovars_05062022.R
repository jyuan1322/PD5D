library(Seurat)
library(reticulate)
use_condaenv("r_leidenalg", required=T)
library(hdf5r)
library(SeuratWrappers)
library(harmony)
library(data.table)
library(glmpca)
library(ggplot2)

# remove duplicate columns from a sparse matrix
# source: https://stackoverflow.com/a/51457395
duplicated.dgCMatrix <- function (dgCMat, MARGIN, include.all.zero.vectors = TRUE) {
    MARGIN <- as.integer(MARGIN)
    J <- rep(1:ncol(dgCMat), diff(dgCMat@p))
    I <- dgCMat@i + 1
    x <- dgCMat@x
    if (MARGIN == 1L) {
        ## check duplicated rows
        names(x) <- J
        if (include.all.zero.vectors) {
            RowLst <- split(x, factor(I, levels = 1:nrow(dgCMat)))
        } else {
            RowLst <- split(x, I)  ## will do `factor(I)` internally in `split`
        }
        result <- duplicated.default(RowLst)
    } else if (MARGIN == 2L) {
        ## check duplicated columns
        names(x) <- I
        if (include.all.zero.vectors) {
            ColLst <- split(x, factor(J, levels = 1:ncol(dgCMat)))
        } else {
            ColLst <- split(x, J)  ## will do `factor(J)` internally in `split`
        }
        result <- duplicated.default(ColLst)
    } else {
        warning("invalid MARGIN; return NULL")
        result <- NULL
    }
    result
}

# remove variablefeatures if they appear too rarely in the spatial samples
# pass_genes <- filter_variablefeatures(combined_controls, VariableFeatures(combined_controls), nzero_spot_thresh=20, frac_sample_thresh=0.3, bin_size=50)
filter_variablefeatures <- function(comb_seurat, gene_list, nzero_spot_thresh=20,
                               frac_sample_thresh=0.3, bin_size=50) {
    num_samples <- length(unique(comb_seurat@meta.data$orig.ident))
    sample_thresh <- floor(num_samples * frac_sample_thresh)
    print(paste("Genes must be nonzero in >", nzero_spot_thresh,
                " spots in >", sample_thresh, " samples", sep=""))
    sample_ids <- comb_seurat@meta.data$orig.ident

    pass_genes <- c()
    ii <- 0
    
    for (gene_chunk in split(gene_list, ceiling(seq_along(gene_list)/bin_size))) {
        gene_matrix <- comb_seurat@assays$Spatial@counts[gene_chunk, ]
        gene_matrix_gt0 <- gene_matrix > 0
        for (row in 1:nrow(gene_matrix_gt0)) {
            gene <- rownames(gene_matrix_gt0)[row]
            idxs <- gene_matrix_gt0[row, ] # slow step
            nzero_spots <- sample_ids[idxs]
            nzero_spots <- table(nzero_spots)
            pass_samples <- names(nzero_spots)[nzero_spots > nzero_spot_thresh]
            ii <- ii+1
            if (length(pass_samples) > sample_thresh) {
                pass_genes <- c(pass_genes, gene)
                print(paste(length(pass_genes), "/", ii, "Passed:", gene, sep=" "))
            } else {
                print(paste(length(pass_genes), "/", ii, "Failed:", gene, sep=" "))
            }
        }
    }
    return(pass_genes)
}

cluster_pipeline <- function(comb_seurat, job_name,
                        harmony_vars = c("orig.ident", "batch", "sex", "expired_age"),
                        harmony_thetas = c(0.4, 0.4, 0.4, 0.4),
                        resolutions_list = c(0.5, 0.8),
                        random_seed = 101,
                        remove_mito_ribo = FALSE) {
    dir.create(job_name)

    checkpoints_file <- file.path(job_name, "checkpoints.txt")
    write(paste(Sys.time(), job_name),file=checkpoints_file,append=FALSE)
    write(paste(Sys.time(), "Harmony covars:"),file=checkpoints_file,append=TRUE)
    write(harmony_vars,file=checkpoints_file,append=TRUE)
    write(harmony_thetas,file=checkpoints_file,append=TRUE)
    write(paste(Sys.time(), "random seed:", random_seed),file=checkpoints_file,append=TRUE)
    write(paste(Sys.time(), "cluster res:"),file=checkpoints_file,append=TRUE)
    write(resolutions_list,file=checkpoints_file,append=TRUE)
    
    comb_seurat <- NormalizeData(comb_seurat,
                       normalization.method = "LogNormalize",
                       scale.factor = 10000)
    comb_seurat <- FindVariableFeatures(comb_seurat,
                       selection.method = "vst",
                       nfeatures = 4000)
    comb_seurat <- ScaleData(comb_seurat,
                       features = rownames(comb_seurat))

    gene_list <- VariableFeatures(comb_seurat)
    pass_genes <- filter_variablefeatures(comb_seurat, gene_list,
                                nzero_spot_thresh=20,
                                frac_sample_thresh=0.3, bin_size=50)
    # almost no intersection here?
    # if (remove_mito_ribo) {
    #     mito_genes <- rownames(comb_seurat)[grep("^MT-",rownames(comb_seurat))]
    #     ribo_genes <- rownames(comb_seurat)[grep("^RP[SL]",rownames(comb_seurat))]
    # }
    VariableFeatures(comb_seurat) <- pass_genes

    # filter out spots that are all-zero in variable genes, or duplicates
    varfeat_counts <- comb_seurat@assays$Spatial@counts[VariableFeatures(comb_seurat), ]
    dup_spots <- which(colSums(varfeat_counts) == 0 | duplicated.dgCMatrix(varfeat_counts, 2))
    total_zero_count <- length(which(colSums(varfeat_counts) == 0))
    print(sprintf("Variable features passing filtering: %s", length(pass_genes)))
    write(sprintf("Variable features passing filtering: %s", length(pass_genes)),file=checkpoints_file,append=TRUE)
    print(sprintf("Total spots all zero in filtered variable features: %s", total_zero_count))
    write(sprintf("Total spots all zero in filtered variable features: %s", total_zero_count),file=checkpoints_file,append=TRUE)
    print(sprintf("Total dup spots removed: %s", length(dup_spots)))
    write(sprintf("Total dup spots removed: %s", length(dup_spots)),file=checkpoints_file,append=TRUE)
    if(length(dup_spots) > 0) {
        comb_seurat <- comb_seurat[, -dup_spots]
    }
    
    # To save memory, remove all genes which are not in VariableFeatures
    # comb_seurat <- subset(comb_seurat, features = VariableFeatures(comb_seurat))
    # gc(verbose = TRUE)

    # comb_seurat <- RunPCA(comb_seurat, pc.genes = comb_seurat@var.genes, npcs = 30)
    # eventually, you should switch over, but for now, confirm PCA still works
    set.seed(random_seed)
    print("Running GLMPCA")
    comb_seurat <- RunGLMPCA(comb_seurat,
                       features = pass_genes, L = 30,
                       minibatch = "stochastic", batch_size = 10000)
    # select number of PCs
    # https://stats.stackexchange.com/q/254592
    # https://github.com/willtownes/glmpca/issues/32#issuecomment-859748250
    eigs <- comb_seurat@reductions$glmpca@stdev^2
    prop_variance <- eigs / sum(eigs)
    PC <- 1:length(prop_variance)

    pdf(file.path(job_name, "glmpca_out.pdf"))
    ve_pt <- ggplot(data.frame(PC, prop_variance), aes(x=PC, y=prop_variance)) + geom_line()
    print(ve_pt)
    dm_pt <- DimPlot(object = comb_seurat, reduction = "glmpca",
                 pt.size = .1, group.by = "orig.ident") + NoLegend()
    print(dm_pt)
    vn_pt <- VlnPlot(object = comb_seurat, features = "GLMPC_1",
                 group.by = "orig.ident", pt.size = .1) + NoLegend()
    print(vn_pt)
    dev.off()
    
    # combined <- RunHarmony(combined, group.by.vars = c("orig.ident"), reduction = "pca", plot_convergence = TRUE, theta = c(1))
    set.seed(random_seed)
    print("Running Harmony")
    pdf(file.path(job_name, "harmony_out.pdf"))
    comb_seurat <- RunHarmony(comb_seurat, reduction.use = "glmpca",
                       plot_convergence = TRUE,
                       max_iter = 20,
                       assay.use = "Spatial",
                       group.by.vars = harmony_vars,
                       theta = harmony_thetas,
                       lambda = rep(1, length(harmony_thetas)))
    dm_pt <- DimPlot(object = comb_seurat, reduction = "harmony",
                 pt.size = .1, group.by = "orig.ident") + NoLegend()
    print(dm_pt)
    vn_pt <- VlnPlot(object = comb_seurat, features = "harmony_1",
                 group.by = "orig.ident", pt.size = .1) + NoLegend()
    print(vn_pt)
    dev.off()

    # The last value for resolution is what is stored in seurat_clusters
    # Use Spatial_snn_res.X for clustering
    comb_seurat <- comb_seurat %>%
        RunUMAP(reduction = "harmony", dims = 1:30) %>%
        FindNeighbors(reduction = "harmony", dims = 1:30) %>%
        identity()
    for (res in resolutions_list) {
        set.seed(random_seed)
        print(paste("Finding Clusters with res=", res, sep=''))
        column_name <- paste("Spatial_snn_res.", res, sep='')
        # comb_seurat <- comb_seurat %>%
        #     RunUMAP(reduction = "harmony", dims = 1:30) %>%
        #     FindNeighbors(reduction = "harmony", dims = 1:30) %>%
        #     FindClusters(resolution = res, algorithm=4, method="igraph") %>%
        #     identity()
        comb_seurat <- comb_seurat %>%
            FindClusters(resolution = res, algorithm=4, method="igraph") %>%
            identity()

        pdf(file.path(job_name,
                      paste("leiden_out_res", res, ".pdf", sep='')))
        dm_pt <- DimPlot(comb_seurat, reduction = "umap",
                     label = TRUE, pt.size = .1,
                     group.by = column_name) + ggtitle(column_name)
        print(dm_pt)
        dev.off()

        pdf(file.path(job_name,
                      paste("spatial_mapped_res", res, ".pdf", sep='')))
        for (img in names(comb_seurat@images)) {
            sp_pt <- SpatialPlot(comb_seurat, group.by= column_name, image.alpha=0.3, images=img)
            print(sp_pt)
        }
        dev.off()
    }

    # Save the Seurat object
    # Regenerate data and scale.data upon reloading
    comb_diet <- DietSeurat(comb_seurat, counts=TRUE, data=TRUE,
                     scale.data=FALSE, dimreducs = c('glmpca', 'harmony', 'umap'))
    saveRDS(comb_diet, file.path(job_name, "comb_seurat_diet_out.rds"))
    return(comb_seurat)
}
