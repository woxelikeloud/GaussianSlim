DATASET_ROOT="dataset/deepblending"
CODE_ROOT="GaussianSlim/recon"

# 定义指定的数据集列表
datalist=(
    "Lumber_Reconstruction_Inputs_Outputs"
    # "NightSnow_Reconstruction_Inputs_Outputs"
    # "Playroom_Reconstruction_Inputs_Outputs"
    # "Ponche_Reconstruction_Inputs_Outputs"
    # "SaintAnne_Reconstruction_Inputs_Outputs"
    # "Shed_Reconstruction_Inputs_Outputs"
    # "Tree-18_Reconstruction_Inputs_Outputs"
    # "Yellowhouse-12_Reconstruction_Inputs_Outputs"
)

for dir in "$DATASET_ROOT"/*/; do
    scene_name=$(basename "$dir")
# for scene_name in "${datalist[@]}"; do
#     dir="$DATASET_ROOT/$scene_name"
    
    if [ ! -d "$dir" ]; then
        echo "跳过: 目录 $dir 不存在"
        continue
    fi

    echo "------------------------------------------------"
    echo "正在处理数据集: $scene_name"
    

    SOURCE_PATH="$dir"

    if [ ! -d "$SOURCE_PATH/colmap/sparse" ]; then
        echo "提示: 在 $SOURCE_PATH 下未找到 sparse 文件夹"
    fi
    
    echo "使用源路径: $SOURCE_PATH"
    echo "------------------------------------------------"
    
    # 3. 执行训练
    CUDA_VISIBLE_DEVICES=0 python "$CODE_ROOT/train.py" -s "$SOURCE_PATH/colmap" -m "$CODE_ROOT/output/deepblending_eval_num/$scene_name" --eval --disable_viewer
done
