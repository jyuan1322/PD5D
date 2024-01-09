# ST BayesSpace layer detection
# This script is based on the tutorial example:
# https://edward130603.github.io/BayesSpace/articles/maynard_DLPFC.html

library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(scuttle)
library(optparse)

option_list = list(
  make_option(c("-i", "--input"), type="character", default=NULL, 
              help="input 10x Visium dir (outs/)", metavar="character"),
  make_option(c("-o", "--output"), type="character", default="bayespace_out.png", 
              help="output png file name [default= %default]", metavar="character"),
  make_option(c("-t", "--title"), type="character", default="BayesSpace", 
              help="sample display name [default= %default]", metavar="character")
);
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$input)){
  print_help(opt_parser)
  stop("input dir is required", call.=FALSE)
}


set.seed(101)
sce <- readVisium(opt$input)
# https://github.com/edward130603/BayesSpace/issues/28
sce <- sce[, colSums(counts(sce)) > 0]
sce <- logNormCounts(sce)

dec <- scran::modelGeneVar(sce)
top <- scran::getTopHVGs(dec, n=2000)
dlpfc <- scater::runPCA(sce, subset_row=top)
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)
q<-7
d<-15
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium', nrep=50000, gamma=3, save.chain=TRUE)

# saveRDS(dlpfc, file = "bayesspace_dlpfc_out.rds")
saveRDS(dlpfc, file = paste(opt$title, "rds", sep="."))

# rotate image 180 degrees
colData(dlpfc)[,'row'] = max(colData(dlpfc)[,'row']) - colData(dlpfc)[,'row']
colData(dlpfc)[,'col'] = max(colData(dlpfc)[,'col']) - colData(dlpfc)[,'col']

labels <- dplyr::recode(dlpfc$spatial.cluster, 1,2,3,4,5,6,7)
# png('rplot.png')
png(opt$output)
clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  scale_fill_viridis_d(option = "A", labels = 1:7) +
  labs(title=opt$title)
dev.off()
