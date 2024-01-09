Clustering and analysis pipeline for Visium 10X spatial transcriptomic data
Jie Yuan 1/8/2024

Analysis steps are split into numbered folders.


1_sample_QC
- xianjun_qc_all_data_01272022.R: Flag sample outliers for removal by aggregating per-sample count matrices and clustering.
- post_qc_all_data_to_seurat_10162023.R: Combine all sample count matrices and metadata into a single Seurat object.

2_clustering_pipeline
- spatial_pipeline_filter_varfeat_duplicates_morecovars_05062022.R: Runs the main clustering pipeline. This includes the following:
    - variable feature selection
    - clustering using GLM-PCA (2019)
    - batch correction using Harmony (2019)
    - Neighborhood detection using Leiden algorithm scanning across resolution values
- run_spatial_pipeline_V4_10272023.R: Runs the clustering pipeline specified in the above script.

3_post_clustering_validation
- validate_best_layers_by_resolution_06302022.R: In middle temporal gyrus, compares cluster annotations across resolution values to cortical layer spatial data from Maynard et al., 2021. A permutation function identifies the best cluster-to-layer assignment for each resolution.
- validate_new_layer4_12132022.R: Layer 4 is extracted by sub-clustering Layer 5 and extracting one of its sub-clusters. This reruns the validation with the new Layer 4 annotation.

4_pseudobulk_DE
- pseudobulk_DE_DESeq2_V4_12052023.R: Performs pseudobulk DE analysis using DESeq2 on each cortical layer separately.
