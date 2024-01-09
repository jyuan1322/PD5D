source("./spatial_pipeline_filter_varfeat_duplicates_morecovars_05062022.R")

# 10/23/2023 5 covariates
# merged_seurat <- "/mnt/data0/projects/ASAP/ST/data/MTG_visium/comb_seurat_V2_postqc_10162023.rds"
# new server
# merged_seurat <- "/mnt/data/projects/ASAP/data/comb_seurat_V2_postqc_10182023.rds"
# merged_seurat <- "/mnt/data0/projects/ASAP/ST/data/MTG_visium/comb_seurat_V3_postqc_10182023.rds"
merged_seurat <- "/mnt/data0/projects/ASAP/ST/data/MTG_visium/comb_seurat_V4_postqc_10272023.rds"
combined <- readRDS(merged_seurat)
resolutions_list <- c(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
harmony_vars <- c("orig.ident", "sex", "expired_age", "PMI", "RIN")
harmony_thetas <- c(0.4, 0.4, 0.4, 0.4, 0.4)
for(random_seed in c(101, 201, 301)) {
    combined_looprun <- combined
    job_name <- sprintf("all_data_5covar0p4_V4_seed%s_10272023", random_seed)
    cluster_pipeline(combined_looprun, job_name,
                     harmony_vars = harmony_vars,
                     harmony_thetas = harmony_thetas,
                     resolutions_list = resolutions_list,
                     random_seed = random_seed)
}

# # 10/23/2023 only sample_id in Harmony covariates
# harmony_vars <- c("orig.ident")
# harmony_thetas <- c(2.0)
# for(random_seed in c(101, 201, 301)) {
#     combined_looprun <- combined
#     job_name <- sprintf("all_data_1covar2p0_V3_seed%s_10182023", random_seed)
#     cluster_pipeline(combined, job_name,
#                      harmony_vars = harmony_vars,
#                      harmony_thetas = harmony_thetas,
#                      resolutions_list = resolutions_list,
#                      random_seed = random_seed)
# }
