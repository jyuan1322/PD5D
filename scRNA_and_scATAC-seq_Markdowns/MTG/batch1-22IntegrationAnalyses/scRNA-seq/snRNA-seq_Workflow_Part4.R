set.seed(100)

library(Seurat)
library(cowplot)
library(ggplot2)
library(ggsci)
library(dplyr)
library(psych)
library(pheatmap)
library(harmony)
#library(clusterProfiler)
#library(org.Hs.eg.db)
library(DOSE)
library(GOSemSim)
library(enrichplot)
library(sciplot)
library(reshape2)
library(MAST)
library(Matrix.utils)
library(optparse)
library(stats)
library(fgsea)

option_list = list(
  make_option(c("-f", "--SeuratObject"), type="character", default=NULL, 
              help="Path to Seurat Object", metavar="character"),
    make_option(c("-c", "--ClusterIdent"), type="character", default="NULL", 
              help="Cluster Identity", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

SeuratObjectPath <- opt$SeuratObject

Cluster_Ident <- opt$ClusterIdent

Cluster_Ident

SeuratObject <- readRDS(SeuratObjectPath)

SeuratObject <- RenameIdents(SeuratObject, `1` = "GLU_Neurons", `2` = "GLU_Neurons", `3` = "Oligodendrocytes", `4` = "GLU_Neurons", `5` = "Astrocytes", `6` = "GLU_Neurons", `7` = "GLU_Neurons", `8` = "GABA_Neurons",`9` = "GABA_Neurons",`10` = "GLU_Neurons", `11` = "Microglia",`12` = "GABA_Neurons",`13` = "Unknown_Cluster_13",`14` = "OPCs",`15` = "GLU_Neurons", `16`= "Cajal_Retzius_Cells", `17`="GLU_GABA_Neurons", `18`="GABA_Neurons", `19`="GABA_Neurons",  `20`="Endothelial", `21` = "Endothelial", `22` = "Unknown_Cluster_22", `23` = "GLU_Neurons", `24` = "GLU_Neurons", `25` = "GLU_Neurons", `26` = "GLU_Neurons",`27` = "GABA_Neurons",`28` = "Unknown_Cluster_28",`29` = "CD8+_T_Cells",`30` = "Unknown_Cluster_30")

C2_Gene_Sets <- gmtPathways("~/databases/GSEA_Tables/c2.all.v7.4.symbols.gmt")

ID_Order <- unique(SeuratObject@meta.data$sample_id)

SeuratObject@meta.data$grouped_case <- SeuratObject@meta.data$case

SeuratObject@meta.data$grouped_case <- gsub("PD|ILB","PD_and_ILB",SeuratObject@meta.data$grouped_case)

mkfilter <- function(cmatrixline) {
        sum(cmatrixline > 0)/length(cmatrixline)*100
} 

Ident_Object <- subset(SeuratObject, idents = Cluster_Ident)
Cluster_Ident
Ident_Object <- Ident_Object[apply(Ident_Object@assays$RNA@counts,1,mkfilter) >= 20,]
Ident_Object@meta.data$DetRate <- as.vector(scale(colSums(Ident_Object@assays$RNA@counts)))
Idents(Ident_Object) <- "case"
FilteredGeneCount <- dim(Ident_Object)[1]

rm(SeuratObject)

ClusterfGSEA <- function(ClusterIdent, IdentObj, ident1, ident2){
  IdentObj@meta.data$DetRate <- as.vector(scale(colSums(IdentObj@assays$RNA@counts > 0)))
  IdentObj.Markers <- FindMarkers(IdentObj, ident.1 = ident1, ident.2 = ident2, verbose = FALSE, test.use = "MAST", latent.vars = c("sex","DetRate","batch","age","PMI","RIN"), logfc.threshold = 0, min.pct = 0)
  IdentObj.Markers$gene <- rownames(IdentObj.Markers)
  IdentObj.Avg <- as.data.frame(AverageExpression(IdentObj, verbose = FALSE)$RNA)
  IdentObj.Avg$gene <- rownames(IdentObj.Avg)
  ProgMarkersIdentObj.Avg <- IdentObj.Avg[IdentObj.Avg$gene %in% unique(IdentObj.Markers$gene),]
  ProgMarkersIdentObj.Avg <- ProgMarkersIdentObj.Avg[match(IdentObj.Markers$gene,ProgMarkersIdentObj.Avg$gene),]
  generanks <- IdentObj.Markers$avg_log2FC
  names(generanks) <- rownames(IdentObj.Markers)
  generanks <- generanks[order(generanks, decreasing = TRUE)]
  fgseaRes = fgsea(C2_Gene_Sets, stats=generanks, minSize=10, maxSize=Inf, nPermSimple=10000)
  fgseaResFilter <- fgseaRes[fgseaRes$padj <= 0.05,]
  fgseaResFilter$leadingEdge = vapply(fgseaResFilter$leadingEdge, paste, collapse = ", ", character(1L))
  if (nrow(fgseaResFilter) > 0) {
  write.table(fgseaResFilter, file = paste("Files/GSEA/",ClusterIdent,"_",ident1,"_vs_",ident2,"_GSEA_Sig_Genesets.tsv",sep = ""), quote = FALSE, col.names = TRUE, row.names = FALSE, sep = "\t")
  }
  IdentObj.Markers$ident1_mean <- ProgMarkersIdentObj.Avg[[ident1]]
  IdentObj.Markers$ident2_mean <- ProgMarkersIdentObj.Avg[[ident2]]
  colnames(IdentObj.Markers)[colnames(IdentObj.Markers) == "ident1_mean"] <- paste(ident1,"_mean",sep="")
  colnames(IdentObj.Markers)[colnames(IdentObj.Markers) == "ident2_mean"] <- paste(ident2,"_mean",sep="")
  IdentObj.Markers$Status <- "Upregulated"
  IdentObj.Markers$Status[IdentObj.Markers$avg_log2FC < 0] <- "Downregulated"
  IdentObj.Markers$Status <- factor(IdentObj.Markers$Status, c("Upregulated","Downregulated"))
  IdentObj.Markers$FilteredGeneCount <- FilteredGeneCount
  IdentObj.Markers$BF_p_val <- 0.05/unique(IdentObj.Markers$FilteredGeneCount)
  write.csv(IdentObj.Markers, file = paste("Files/DE_Genes/AllGenes_",ClusterIdent,"_Markers_",ident1,"_vs_",ident2,".csv",sep = ""), quote = FALSE)
}

#PD vs HC
Ident.PDvsHC_Prog.Markers <- FindMarkers(Ident_Object, ident.1 = "PD", ident.2 = "HC", verbose = FALSE, test.use = "MAST", latent.vars = c("batch","RIN","PMI","age","sex","DetRate"))
Ident.PDvsHC_Prog.Markers$gene <- rownames(Ident.PDvsHC_Prog.Markers)
Ident.PDvsHC_Prog.Markers <-
        Ident.PDvsHC_Prog.Markers[Ident.PDvsHC_Prog.Markers$p_val_adj <= 0.05,]

if (length(Ident.PDvsHC_Prog.Markers$gene) != 0) {
        avg.Ident_Object_PDvsHC <- as.data.frame(AverageExpression(Ident_Object, verbose = FALSE)$RNA)
        avg.Ident_Object_PDvsHC$gene <- rownames(avg.Ident_Object_PDvsHC)
        
        avg.Ident_Object_progmarkers_PDvsHC <- avg.Ident_Object_PDvsHC[avg.Ident_Object_PDvsHC$gene %in% unique(Ident.PDvsHC_Prog.Markers$gene),]
        avg.Ident_Object_progmarkers_PDvsHC <- avg.Ident_Object_progmarkers_PDvsHC[match(Ident.PDvsHC_Prog.Markers$gene,avg.Ident_Object_progmarkers_PDvsHC$gene),]
        Ident.PDvsHC_Prog.Markers$PD_mean <- avg.Ident_Object_progmarkers_PDvsHC$PD
        Ident.PDvsHC_Prog.Markers$HC_mean <- avg.Ident_Object_progmarkers_PDvsHC$HC
        Ident.PDvsHC_Prog.Markers$Status <- "Upregulated"
        Ident.PDvsHC_Prog.Markers$Status[Ident.PDvsHC_Prog.Markers$avg_log2FC < 0] <- "Downregulated"
        Ident.PDvsHC_Prog.Markers$Status <- factor(Ident.PDvsHC_Prog.Markers$Status, c("Upregulated","Downregulated"))
        Ident.PDvsHC_Prog.Markers$FilteredGeneCount <- FilteredGeneCount
        Ident.PDvsHC_Prog.Markers$BF_p_val <- 0.05/unique(Ident.PDvsHC_Prog.Markers$FilteredGeneCount)
        write.csv(Ident.PDvsHC_Prog.Markers,file = paste("Files/DE_Genes/All_SigGenes_",Cluster_Ident,"_Markers_HC_vs_PD.csv",sep = ""), quote = FALSE)
}

ClusterfGSEA(Cluster_Ident,Ident_Object,"PD","HC")

# ILB vs HC


Ident.ILBvsHC_Prog.Markers <- FindMarkers(Ident_Object, ident.1 = "ILB", ident.2 = "HC", verbose = FALSE, test.use = "MAST", latent.vars = c("batch","RIN","PMI","age","sex","DetRate"))
Ident.ILBvsHC_Prog.Markers$gene <- rownames(Ident.ILBvsHC_Prog.Markers)
Ident.ILBvsHC_Prog.Markers <- Ident.ILBvsHC_Prog.Markers[Ident.ILBvsHC_Prog.Markers$p_val_adj <= 0.05,]

if (length(Ident.ILBvsHC_Prog.Markers$gene) != 0){
        avg.Ident_Object_ILBvsHC <- as.data.frame(AverageExpression(Ident_Object, verbose = FALSE)$RNA)
        avg.Ident_Object_ILBvsHC$gene <- rownames(avg.Ident_Object_ILBvsHC)
        
        avg.Ident_Object_progmarkers_ILBvsHC <- avg.Ident_Object_ILBvsHC[avg.Ident_Object_ILBvsHC$gene %in% unique(Ident.ILBvsHC_Prog.Markers$gene),]
        avg.Ident_Object_progmarkers_ILBvsHC <- avg.Ident_Object_progmarkers_ILBvsHC[match(Ident.ILBvsHC_Prog.Markers$gene,avg.Ident_Object_progmarkers_ILBvsHC$gene),]
        Ident.ILBvsHC_Prog.Markers$ILB_mean <- avg.Ident_Object_progmarkers_ILBvsHC$ILB
        Ident.ILBvsHC_Prog.Markers$HC_mean <- avg.Ident_Object_progmarkers_ILBvsHC$HC
        Ident.ILBvsHC_Prog.Markers$Status <- "Upregulated"
        Ident.ILBvsHC_Prog.Markers$Status[Ident.ILBvsHC_Prog.Markers$avg_log2FC < 0] <- "Downregulated"
        Ident.ILBvsHC_Prog.Markers$Status <- factor(Ident.ILBvsHC_Prog.Markers$Status, c("Upregulated","Downregulated"))
        Ident.ILBvsHC_Prog.Markers$FilteredGeneCount <- FilteredGeneCount
        Ident.ILBvsHC_Prog.Markers$BF_p_val <- 0.05/unique(Ident.ILBvsHC_Prog.Markers$FilteredGeneCount)        
        write.csv(Ident.ILBvsHC_Prog.Markers,file = paste("Files/DE_Genes/All_SigGenes_",Cluster_Ident,"_Markers_HC_vs_ILB.csv",sep = ""), quote = FALSE)
}

ClusterfGSEA(Cluster_Ident,Ident_Object,"ILB","HC")

# PD vs ILB

Ident.PDvsILB_Prog.Markers <- FindMarkers(Ident_Object, ident.1 = "PD", ident.2 = "ILB", verbose = FALSE, test.use = "MAST", latent.vars = c("batch","RIN","PMI","age","sex","DetRate"))
Ident.PDvsILB_Prog.Markers$gene <- rownames(Ident.PDvsILB_Prog.Markers)
Ident.PDvsILB_Prog.Markers <-
        Ident.PDvsILB_Prog.Markers[Ident.PDvsILB_Prog.Markers$p_val_adj <= 0.05,]

if (length(Ident.PDvsILB_Prog.Markers$gene) != 0) {
        avg.Ident_Object_PDvsILB <- as.data.frame(AverageExpression(Ident_Object, verbose = FALSE)$RNA)
        avg.Ident_Object_PDvsILB$gene <- rownames(avg.Ident_Object_PDvsILB)
        
        avg.Ident_Object_progmarkers_PDvsILB <- avg.Ident_Object_PDvsILB[avg.Ident_Object_PDvsILB$gene %in% unique(Ident.PDvsILB_Prog.Markers$gene),]
        avg.Ident_Object_progmarkers_PDvsILB <- avg.Ident_Object_progmarkers_PDvsILB[match(Ident.PDvsILB_Prog.Markers$gene,avg.Ident_Object_progmarkers_PDvsILB$gene),]
        Ident.PDvsILB_Prog.Markers$PD_mean <- avg.Ident_Object_progmarkers_PDvsILB$PD
        Ident.PDvsILB_Prog.Markers$ILB_mean <- avg.Ident_Object_progmarkers_PDvsILB$ILB
        Ident.PDvsILB_Prog.Markers$Status <- "Upregulated"
        Ident.PDvsILB_Prog.Markers$Status[Ident.PDvsILB_Prog.Markers$avg_log2FC < 0] <- "Downregulated"
        Ident.PDvsILB_Prog.Markers$Status <- factor(Ident.PDvsILB_Prog.Markers$Status, c("Upregulated","Downregulated"))
        Ident.PDvsILB_Prog.Markers$FilteredGeneCount <- FilteredGeneCount
        Ident.PDvsILB_Prog.Markers$BF_p_val <- 0.05/unique(Ident.PDvsILB_Prog.Markers$FilteredGeneCount)        
        write.csv(Ident.PDvsILB_Prog.Markers,file = paste("Files/DE_Genes/All_SigGenes_",Cluster_Ident,"_Markers_ILB_vs_PD.csv",sep = ""), quote = FALSE)
}

ClusterfGSEA(Cluster_Ident,Ident_Object,"PD","ILB")


Idents(Ident_Object) <- "grouped_case"


Ident.PD_and_ILBvsHC_Prog.Markers <- FindMarkers(Ident_Object, ident.1 = "PD_and_ILB", ident.2 = "HC", verbose = FALSE, test.use = "MAST", latent.vars = c("batch","RIN","PMI","age","sex","DetRate"))
Ident.PD_and_ILBvsHC_Prog.Markers$gene <- rownames(Ident.PD_and_ILBvsHC_Prog.Markers)
Ident.PD_and_ILBvsHC_Prog.Markers <- Ident.PD_and_ILBvsHC_Prog.Markers[Ident.PD_and_ILBvsHC_Prog.Markers$p_val_adj <= 0.05,]

if (length(Ident.PD_and_ILBvsHC_Prog.Markers$gene) != 0){
        avg.Ident_Object_PD_and_ILBvsHC <- as.data.frame(AverageExpression(Ident_Object, verbose = FALSE)$RNA)
        avg.Ident_Object_PD_and_ILBvsHC$gene <- rownames(avg.Ident_Object_PD_and_ILBvsHC)
        
        avg.Ident_Object_progmarkers_PD_and_ILBvsHC <- avg.Ident_Object_PD_and_ILBvsHC[avg.Ident_Object_PD_and_ILBvsHC$gene %in% unique(Ident.PD_and_ILBvsHC_Prog.Markers$gene),]
        avg.Ident_Object_progmarkers_PD_and_ILBvsHC <- avg.Ident_Object_progmarkers_PD_and_ILBvsHC[match(Ident.PD_and_ILBvsHC_Prog.Markers$gene,avg.Ident_Object_progmarkers_PD_and_ILBvsHC$gene),]
        Ident.PD_and_ILBvsHC_Prog.Markers$ILB_mean <- avg.Ident_Object_progmarkers_PD_and_ILBvsHC$PD_and_ILB
        Ident.PD_and_ILBvsHC_Prog.Markers$HC_mean <- avg.Ident_Object_progmarkers_PD_and_ILBvsHC$HC
        Ident.PD_and_ILBvsHC_Prog.Markers$Status <- "Upregulated"
        Ident.PD_and_ILBvsHC_Prog.Markers$Status[Ident.PD_and_ILBvsHC_Prog.Markers$avg_log2FC < 0] <- "Downregulated"
        Ident.PD_and_ILBvsHC_Prog.Markers$Status <- factor(Ident.PD_and_ILBvsHC_Prog.Markers$Status, c("Upregulated","Downregulated"))
        Ident.PD_and_ILBvsHC_Prog.Markers$FilteredGeneCount <- FilteredGeneCount
        Ident.PD_and_ILBvsHC_Prog.Markers$BF_p_val <- 0.05/unique(Ident.PD_and_ILBvsHC_Prog.Markers$FilteredGeneCount)
        write.csv(Ident.PD_and_ILBvsHC_Prog.Markers,file = paste("Files/DE_Genes/All_SigGenes_",Cluster_Ident,"_Markers_HC_vs_PD_and_ILB.csv",sep = ""), quote = FALSE)
}

ClusterfGSEA(Cluster_Ident,Ident_Object,"PD_and_ILB","HC")



