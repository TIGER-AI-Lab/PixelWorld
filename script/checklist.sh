#!/usr/bin/env bash

# 定义数组
DATASETS=("SuperGLUEDataset" "MMLUProDataset" "ARCDataset" "GLUEDataset" "GSM8KDataset" "MBPPDataset" "TableBenchDataset")
MODES=("text" "img")
MODEL_NAME="QWen2VLWrapper"

# 与 Python 字典对应的映射 (可在bash里直接硬编码写成case或关联数组)
# 这里演示用关联数组，需要bash 4+
declare -A DATASET_PROMPT_CORRESPONDENCE
DATASET_PROMPT_CORRESPONDENCE["SuperGLUEDataset"]="base"
DATASET_PROMPT_CORRESPONDENCE["GLUEDataset"]="base"
DATASET_PROMPT_CORRESPONDENCE["MMLUProDataset"]="cot"
DATASET_PROMPT_CORRESPONDENCE["ARCDataset"]="base"
DATASET_PROMPT_CORRESPONDENCE["MBPPDataset"]="base"
DATASET_PROMPT_CORRESPONDENCE["GSM8KDataset"]="cot"
DATASET_PROMPT_CORRESPONDENCE["TableBenchDataset"]="base"

# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 只循环 DATASET 和 MODE；PROMPT 从映射里获取
for DATASET in "${DATASETS[@]}"; do
  # 获取当前 dataset 对应 prompt
  PROMPT="${DATASET_PROMPT_CORRESPONDENCE[$DATASET]}"

  # 如果有可能找不到对应关系，可以加个判断
  if [ -z "$PROMPT" ]; then
    echo "Warning: No prompt mapping found for $DATASET, skip..."
    continue
  fi

  for MODE in "${MODES[@]}"; do
      # 构建日志文件
      LOG_FILE="$LOG_DIR/${DATASET}_${MODEL_NAME}_${MODE}_${PROMPT}.log"
      # 构建命令
      COMMAND="python data.py --dataset $DATASET --model $MODEL_NAME --mode $MODE --prompt $PROMPT"

      # 执行
      echo "Running: $COMMAND"
      nohup $COMMAND > "$LOG_FILE" 2>&1

      # 等待当前任务结束再进行下一个
      wait
  done
done

echo "All tasks completed."
