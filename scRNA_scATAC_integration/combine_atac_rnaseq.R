# Load libraries
library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(EnsDb.Hsapiens.v75)
library(ggplot2)
library(patchwork)
library(hdf5r)
set.seed(1234)

# read atacseq files : 1) count_mat, 2) metadata, 3) chrom_assay
# counts1 is a matrix with rows are genomic regions (ex. chr1:631141-631806 ) 
# and columns are samples (ex. AAACGAAAGAACCATA-2 ). the content in the matrix is 
# the nummber of TNFs in each cell in the specific genomic region.

counts <- Read10X_h5(filename = "./atac_merge/filtered_peak_bc_matrix.h5")
chrom_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c(":", "-"),
  genome = 'hg19',
  fragments = './atac_merge/fragments.tsv.gz'
)
metadata <- read.csv(
  file = "./atac_merge/singlecell.csv",
  header = TRUE,
  row.names = 1
)
# Create Seurat object for the atacseq data
atac0 <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = metadata)

# Preprocess atacseq data: 
preprocess_attac <- function(atac) {
  # extract gene annotations from EnsDb
  annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75)
  
  # change to UCSC style since the data was mapped to hg19
  seqlevelsStyle(annotations) <- 'UCSC'
  genome(annotations) <- "hg19"
  
  # add the gene information to the object
  Annotation(atac) <- annotations
  
  # compute nucleosome signal score per cell
  atac <- NucleosomeSignal(object = atac)
  
  # compute TSS enrichment score per cell
  atac <- TSSEnrichment(object = atac, fast = FALSE)
  
  # add blacklist ratio and fraction of reads in peaks
  atac$pct_reads_in_peaks <- atac$peak_region_fragments / atac$passed_filters * 100
  atac$blacklist_ratio <- atac$blacklist_region_fragments / atac$peak_region_fragments
  
  atac$high.tss <- ifelse(atac$TSS.enrichment > 2, 'High', 'Low')
  TSSPlot(atac, group.by = 'high.tss') + NoLegend()
  
  
  atac$nucleosome_group <- ifelse(atac$nucleosome_signal > 4, 'NS > 4', 'NS < 4')
  FragmentHistogram(object = atac, group.by = 'nucleosome_group')
  return(atac)
}
# Perform preprocessing steps outlined in the preprocess_attac function.
atac0<-preprocess_attac(atac0)
atac0 <- RunTFIDF(atac0)

# Find top binary features for a given assay based on total number of cells containing feature.
# Can specify a minimum cell count, or a lower percentile bound.
dim(atac0@assays$peaks@counts)
atac0 <- FindTopFeatures(atac0, min.cutoff = 'q0')
dim(atac0@assays$peaks@counts)

atac0 <- RunSVD(atac0, n = 50, reduction.name = "svd") # n =  50 by default, saved as lsi
atac0 <- RunLSI(atac0, n = 50, reduction.name = "lsi") # n =  50 by default, saved as lsi
atac0 <- RunUMAP(object = atac0, reduction = 'lsi', dims = 1:30)
atac0 <- FindNeighbors(object = atac0, reduction = 'lsi', dims = 1:30)
atac0 <- FindClusters(object = atac0, verbose = FALSE, algorithm = 1,resolution=0.1) # 3 is SLM algorithm
DimPlot(object = atac0,reduction = 'umap', label = TRUE) + NoLegend()

#summary above : 
# 1) find top 50 features,
# 2) do dimensionality reduction using svd (top 50) or lsi or lsi+umap
# 3) Find Neighbors and Clustering
# 4) Plots

atac0@reductions
saveRDS(atac0,"./atac_merge.rds")
###### 


atac0=readRDS("./atac_merge.rds")

# Compute counts per cell in gene body and promoter region. 
# add the gene activity matrix to the Seurat object as a new assay and normalize it
gene.activities <- GeneActivity(atac0)  
atac0[['RNA']] <- CreateAssayObject(counts = gene.activities)
saveRDS(atac0,"./atac_merge.rds")

#######
atac0=readRDS("./atac_merge.rds")

atac0$nCount_peaks
atac0$nCount_peaks
summary(atac0$nCount_peaks)
 atac1 <- subset(atac0, subset = nCount_peaks >100)


DefaultAssay(atac1) <- 'RNA'
atac1 <- FindVariableFeatures(atac1)

atac1<- NormalizeData(
  object = atac1,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(atac1$nCount_RNA)
)


FeaturePlot(
  object = atac1,
  features = c('MS4A1', 'CD3D', 'LEF1', 'NKG7', 'TREM1', 'LYZ'),
  pt.size = 0.1,
  max.cutoff = 'q95',
  ncol = 3,
  reduction = 'umap'
)

saveRDS(atac0,"./atac_merge.rds")
saveRDS(atac1,"./atac_merge_filtered.rds")
###############
atac1=readRDS("./atac_merge_filtered.rds")
rna=readRDS("single_cell_data/AllMB.rds")

rna@assays$RNA@counts
atac1@assays$RNA@counts
dim(rna@assays$RNA@counts)
dim(atac1@assays$RNA@counts)


# DefaultAssay(atac1) <- "peaks"
# VariableFeatures(atac1) <- names(which(Matrix::rowSums(atac1) > 100))
# atac1 <- RunLSI(atac1, n = 50, scale.max = NULL)
# atac1 <- RunUMAP(atac1, reduction = "lsi", dims = 1:50)
# p1 <- DimPlot(atac1, reduction = "umap") + NoLegend() + ggtitle("scATAC-seq")
# p2 <- DimPlot(rna, label = TRUE, repel = TRUE) + NoLegend() + ggtitle("scRNA-seq")
# p1 + p2


DefaultAssay(atac1)<"RNA"
transfer.anchors <- FindTransferAnchors(
  reference = rna,
  query = atac1,
  features = VariableFeatures(object = atac1), 
  reference.assay = "RNA",
  query.assay = "RNA",
  reduction = 'cca'
)

dim(transfer.anchors@anchors)
transfer.anchors@anchor.features
# rna$celltype <- Idents(rna)

predicted.labels <- TransferData(
  anchorset = transfer.anchors,
  refdata = Idents(rna),
  weight.reduction = atac1[['lsi']],
  dims = 1:30
)

atac1 <- AddMetaData(object = atac1, metadata = predicted.labels)
saveRDS(atac1,"./atac_merge_filtered_matched_100.rds")
###############
rna=readRDS("single_cell_data/AllMB.rds")
atac1=readRDS("./atac_merge_filtered_matched_100.rds")


hist(atac1$prediction.score.max)
abline(v = 0.5, col = "red")

table(atac1$prediction.score.max > 0.1)
atac1.atac.filtered <- subset(atac1, subset = prediction.score.max > 0.1)
atac1.atac.filtered$predicted.id <- factor(atac1.atac.filtered$predicted.id, levels = levels(rna))  # to make 
rna$celltype=Idents(rna)
p1 <- DimPlot(atac1.atac.filtered, group.by = "predicted.id", label = TRUE, repel = TRUE) + ggtitle("scATAC-seq cells") + 
  NoLegend() + scale_colour_hue(drop = FALSE)
p2 <- DimPlot(rna,group.by = "celltype",  label = TRUE, repel = TRUE) + ggtitle("scRNA-seq cells") + 
  NoLegend()

p1 + p2




### Coembedding :

# note that we restrict the imputation to variable genes from scRNA-seq, but could impute the
# full transcriptome if we wanted to
rna$celltype=Idents(rna)
genes.use <- VariableFeatures(rna)
refdata <- GetAssayData(rna, assay = "RNA", slot = "data")[genes.use, ]

# refdata (input) contains a scRNA-seq expression matrix for the scRNA-seq cells.  imputation
# (output) will contain an imputed scRNA-seq matrix for each of the ATAC cells
imputation <- TransferData(anchorset = transfer.anchors, refdata = refdata, weight.reduction = atac1[["lsi"]])

# this line adds the imputed data matrix to the atac object
atac1[["RNA"]] <- imputation
rna$tech="rna"
atac1$tech="atac"
coembed <- merge(x = rna, y = atac1)

# Finally, we run PCA and UMAP on this combined object, to visualize the co-embedding of both
# datasets
coembed <- ScaleData(coembed, features = genes.use, do.scale = FALSE)
coembed <- RunPCA(coembed, features = genes.use, verbose = FALSE)
coembed <- RunUMAP(coembed, dims = 1:30)

coembed$celltype <- ifelse(!is.na(coembed$celltype), coembed$celltype, coembed$predicted.id)
coembed$nFeature_peaks

p1 <- DimPlot(coembed, group.by = "tech")
p2 <- DimPlot(coembed, group.by = "celltype", label = TRUE, repel = TRUE)
p1 + p2

DimPlot(coembed, split.by = "tech", group.by = "celltype", label = TRUE, repel = TRUE) + NoLegend()


coembed$blacklist_region_fragments[is.na(coembed$blacklist_region_fragments)] <- 0
FeaturePlot(coembed, features = "blacklist_region_fragments", max.cutoff = 500)

##########
# change back to working with peaks instead of gene activities
saveRDS(atac1.atac.filtered,"atac1.atac.filtered.rds")

DefaultAssay(atac1.atac.filtered) <- 'peaks'
Idents(atac1.atac.filtered)<-atac1.atac.filtered$predicted.id


da_peaks <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = c("Astrocytes","OPCs"),
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks,"da_peaks.rds")
da_peaks=readRDS("da_peaks.rds")
head(da_peaks)

plot1 <- VlnPlot(
  object = atac1.atac.filtered,
  features = rownames(da_peaks)[1],
  pt.size = 0.1,
  idents = c("dopamine_neuron_potentially_DA2","Astrocytes")
)
plot2 <- FeaturePlot(
  object = atac1.atac.filtered,
  features = rownames(da_peaks)[1],
  pt.size = 0.1
)

plot1 | plot2

fc <- FoldChange(atac1.atac.filtered, ident.1 = "dopamine_neuron_potentially_DA2", ident.2 = c("Astrocytes","OPCs"))
head(fc)

open_DA2 <- rownames(da_peaks[da_peaks$avg_logFC > 0.5, ])
open_Astrocytes <- rownames(da_peaks[da_peaks$avg_logFC < -0.5, ])

closest_genes_DA2 <- ClosestFeature(atac1.atac.filtered, regions = open_DA2)
closest_genes_Astrocytes  <- ClosestFeature(atac1.atac.filtered, regions = open_Astrocytes)

head(closest_genes_DA2)



CoveragePlot(
  object = atac1.atac.filtered,
  region = rownames(da_peaks)[100],
  extend.upstream = 40000,
  extend.downstream = 20000
)

CoverageBrowser(
  object = atac1.atac.filtered,
  region = rownames(da_peaks)[1:10],
  extend.upstream = 40000,
  extend.downstream = 20000
)

library(shiny)

##############

#regions of studies :
atac1.atac.filtered=readRDS("atac1.atac.filtered.rds")
atac1.atac.filtered
unique(Idents(rna))

neurons_master = c('Glu_GABA neurons','GABA_neurons','dopamine_neuron_potentially_DA2')
astrocytes = "Astrocytes"
microglia = "Microglia"
endothelial = "Endothelial cells"
oligo_predominant = c("OPCs","Oligodendrocytes")

#===========
da_peaks_neurons_master <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = neurons_master,
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks_neurons_master ,"da_peaks_neurons_master.rds")
#===========
da_peaks_astrocytes <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = astrocytes,
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks_astrocytes ,"da_peaks_astrocytes.rds")
#===========
da_peaks_microglia <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = microglia,
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks_microglia ,"da_peaks_microglia.rds")
#===========
da_peaks_endothelial <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = endothelial,
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks_endothelial ,"da_peaks_endothelial.rds")
#===========
da_peaks_oligo_predominant <- FindMarkers(
  object = atac1.atac.filtered,
  ident.1 = oligo_predominant,
  ident.2 = NULL,
  min.pct = 0.01,
  test.use = 'LR',
  latent.vars = 'peak_region_fragments'
)
saveRDS(da_peaks_oligo_predominant ,"da_peaks_oligo_predominant.rds")


atac1.atac.filtered=readRDS("atac1.atac.filtered.rds")
library(data.table)
name="neurons_master"

extract_open_regions<-function(names,atac1.atac.filtered,FCthreshold=0.5){
  da_peaks_all=data.frame()
  for (name in names){
    da_peaks=readRDS(paste0("da_peaks_",name,".rds"))
    da_peaks=da_peaks[da_peaks$avg_logFC > FCthreshold, ]
    da_peaks["cluster"]=name
    da_peaks_all=rbind(da_peaks_all,da_peaks)
  }
  da_peaks_all=da_peaks_all%>%rownames_to_column("query_region")
  fwrite(da_peaks_all,"da_peaks_all.csv")
  return(da_peaks_all)
}


extract_open_genes<-function(name,atac1.atac.filtered,FCthreshold=0.5){
  da_peaks=readRDS(paste0("da_peaks_",name,".rds"))
  fwrite(da_peaks ,paste0("da_peaks_",name,".csv"),row.names = TRUE)
  open_da_peaks <- rownames(da_peaks[da_peaks$avg_logFC > FCthreshold, ])
  closest_genes_da_peaks <- ClosestFeature(atac1.atac.filtered, regions = open_da_peaks)
  fwrite(closest_genes_da_peaks ,paste0("closest_genes_da_peaks_",name,".csv"),row.names = TRUE)
  return(closest_genes_da_peaks)
}

extract_open_genes2<-function(name,atac1.atac.filtered,FCthreshold=0.5){
  da_peaks=readRDS(paste0("da_peaks_",name,".rds"))
  #fwrite(da_peaks ,paste0("da_peaks_",name,".csv"),row.names = TRUE)
  open_da_peaks <- rownames(da_peaks[da_peaks$avg_logFC > FCthreshold, ])
  closest_genes_da_peaks <- ClosestFeature(atac1.atac.filtered, regions = open_da_peaks)
  #fwrite(closest_genes_da_peaks ,paste0("closest_genes_da_peaks_",name,".csv"),row.names = TRUE)
  closest_genes_da_peaks$cluster=name
  return(closest_genes_da_peaks)
}


names=c("neurons_master",
       "astrocytes",
       "microglia",
       "endothelial",
       "oligo_predominant")


da_peaks_all=extract_open_regions(names,atac1.atac.filtered,FCthreshold=0.5)
da_peaks_all=fread("da_peaks_all.csv")


open_genes_df_all=data.frame()
  for (name in names) {
  print(name)
  open_genes=extract_open_genes(name,atac1.atac.filtered,FCthreshold=0.5)
  open_genes_df_all=rbind(open_genes_df_all,open_genes)
}



fwrite(open_genes_df_all,"atac_seq_open_genes_df_all.csv")
library(dplyr)
library(tibble)

################################################
open_genes_df_all<-fread('atac_seq_open_genes_df_all.csv')
open_atac_neuron_genes = open_genes_df_all%>%filter(cluster=="neurons_master")%>%pull("gene_name")
rna_cluster_markers=fread("/Users/mnabian/Google Drive/seurat_single_cell/single_cell_data/cluster_markers.csv")


open_atac_neuron_genes = open_genes_df_all%>%filter(cluster=="neurons_master")%>%pull("gene_name")
marker_rna_neuron_genes= rna_cluster_markers%>%
  filter(cluster %in% c('Glu_GABA neurons','GABA_neurons','dopamine_neuron_potentially_DA2'))%>%
  pull("gene")

common_rna_atac_neuron_genes=sort(intersect(open_atac_neuron_genes,marker_rna_neuron_genes))
fwrite(data.frame(rna_atac_neuron_common=common_rna_atac_neuron_genes),"common_rna_atac_neuron_genes.csv")
#############################################
library(data.table)
open_genes_df_all<-fread('atac_seq_open_genes_df_all.csv')
rna_cluster_markers=fread("/Users/mnabian/Google Drive/seurat_single_cell/single_cell_data/cluster_markers.csv")
da_peaks_all=fread("da_peaks_all.csv")
open_atac_neuron_genes = open_genes_df_all%>%filter(cluster=="neurons_master")%>%rename("cluster_atac"="cluster")
length(open_atac_neuron_genes$gene_name)
length(unique(open_atac_neuron_genes$gene_name))


marker_rna_neuron_genes= rna_cluster_markers%>%
  filter(cluster %in% c('Glu_GABA neurons','GABA_neurons','dopamine_neuron_potentially_DA2'))%>%
  rename(gene_name=V1)%>%rename("cluster_rna"="cluster")%>%rename(c("p_val_rna"="p_val",
                                                                    "avg_logFC_rna"="avg_logFC",
                                                                    "p_val_adj_rna"="p_val_adj",
                                                                    "pct.1_rna"="pct.1",
                                                                    "pct.2_rna"="pct.2"))


da_peaks_all=da_peaks_all%>%filter(cluster=="neurons_master")%>%rename(c("p_val_atac"="p_val",
                                                                         "avg_logFC_atac"="avg_logFC",
                                                                         "p_val_adj_atac"="p_val_adj",
                                                                         "pct.1_atac"="pct.1",
                                                                         "pct.2_atac"="pct.2"))
                                                    

common_rna_atac_neuron_genes=open_atac_neuron_genes%>%
  inner_join(marker_rna_neuron_genes,by="gene_name")

common_rna_atac_neuron_genes=common_rna_atac_neuron_genes%>%
  inner_join(da_peaks_all,by="query_region")%>%arrange(desc("p_val_adj_atac"))

sort(table(common_rna_atac_neuron_genes$gene_name),decreasing = TRUE)

fwrite(common_rna_atac_neuron_genes,"common_rna_atac_neuron_genes.csv")
common_rna_atac_neuron_genes
#############################################
common_rna_atac_neuron_genes=fread("common_rna_atac_neuron_genes.csv")
rna_atac_neuron_common=common_rna_atac_neuron_genes$gene_name

rna_atac_neuron_common

install.packages("gprofiler2")
library(gprofiler2)
gostres <- gost(query = common_rna_atac_neuron_genes, 
                organism = "hsapiens", ordered_query = FALSE, 
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE, 
                measure_underrepresentation = FALSE, evcodes = FALSE, 
                user_threshold = 0.05, correction_method = "g_SCS", 
                domain_scope = "annotated", custom_bg = NULL, 
                numeric_ns = "", sources = NULL, as_short_link = FALSE)
fwrite(gostres$result,"geneontology_neuron_genes_rna_atac_common.csv")
################################################


# Todos :   1) Group the clusters make plot for all genes in the table      2) Play with the clusters to make them one. 
  

atac1.atac.filtered_temp=atac1.atac.filtered
unique(Idents(atac1.atac.filtered_temp))
atac1.atac.filtered_temp <- RenameIdents(atac1.atac.filtered_temp, "Glu_GABA neurons" = "neurons_master",
                                        "GABA_neurons" = "neurons_master",
                                        "dopamine_neuron_potentially_DA2" = "neurons_master",
                                        "Oligodendrocytes"="oligodendrocytes_enriched",
                                        "OPCs"="oligodendrocytes_enriched")

atac1.atac.filtered_temp2 <- subset(atac1.atac.filtered_temp, idents = c("neurons_master",
                                                     "oligodendrocytes_enriched",
                                                     "Astrocytes",
                                                     "Microglia",
                                                     "Endothelial cells"))

atac1.atac.filtered_temp@meta.data


#Tn5 insertion frequency
CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "FLT1",
  annotation = TRUE,
  peaks = TRUE
)

atac_ratio_nums=table(Idents(atac1.atac.filtered))/sum(table(Idents(atac1.atac.filtered)))
rna_ratio_nums=table(Idents(rna))/sum(table(Idents(rna)))
atac_ratio_nums=sort(atac_ratio_nums,decreasing = TRUE)
rna_ratio_nums=sort(rna_ratio_nums,decreasing = TRUE)

barplot(rna_ratio_nums,las=2,cex.names=.5)
barplot(atac_ratio_nums,las=2,cex.names=.5)


type(atac_ratio_nums)

barplot(order(rna_ratio_nums,decreasing = TRUE))


Majorcelltype_markergenes <- c("ENO2", "TH", "SLC6A3", "SLC18A2", "SLC17A6", "SLC17A7", "SLC32A1", "GAD1", "GAD2", "AQP4",
                               "GFAP", "PLP1", "OLIG1", "VCAN", "CX3CR1", "P2RY12", "FLT1")

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "MYO9A",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "EPHB1",
  annotation = TRUE,
  window = 100000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "FAM153C",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "FSIP1",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "PPFIA4",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "R3HDM2",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "SCAI",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "CADM3",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE)




for (gene in Majorcelltype_markergenes){
  abc=CoveragePlot(
    object = atac1.atac.filtered_temp2,
    region = gene,
    annotation = TRUE,
    peaks = TRUE
  )
  print(gene)
  #jpeg(paste0(gene,".jpg"))
  ggsave(filename = paste0("./atac_accessibility_markers/",gene,".jpg"), abc)
}

# Plot atac and rna for the following genes :

# UCHL1
# SOX4
# LMX1B 
# SOX6

CoveragePlot(
  object = atac1.atac.filtered_temp2,
  region = "SOX6",
  annotation = TRUE,
  window = 10000,
  peaks = TRUE
)

geneset2=c("SOX4","LMX1B","SOX6","UCHL1")
geneset2=c("SOX6")
for (gene in geneset2){
  print(gene)
  abc=CoveragePlot(
    object = atac1.atac.filtered_temp2,
    region = gene,
    annotation = TRUE,
    window = 10000,
    peaks = TRUE
  )
  print(gene)
  #jpeg(paste0(gene,".jpg"))
  ggsave(filename = paste0("./atac_accessibility_markers/",gene,".jpg"), abc)
}



rna=readRDS("single_cell_data/AllMB.rds")

for (gene in geneset2){
  print(gene)
  abc=VlnPlot(rna, features = gene,pt.size = 0)
  print(gene)
  #jpeg(paste0(gene,".jpg"))
  ggsave(filename = paste0("./rna_expression_markers/",gene,".jpg"), abc)
}


VlnPlot(rna, features = c("SOX4"),pt.size = 0)
VlnPlot(rna, features = c("LMX1B"),pt.size = 0)
VlnPlot(rna, features = c("SOX6"),pt.size = 0)
VlnPlot(rna, features = c("UCHL1"),pt.size = 0)







