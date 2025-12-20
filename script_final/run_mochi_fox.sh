#!/bin/bash
set -e
trap 'echo "出错的命令：$BASH_COMMAND"; export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; bash /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/CTRL-D_training/run_training_2.sh; exit 1' ERR


# create data
# create a new folder in the DyCheck
# create the first output_edited_rended_image folder and cp pointcloud deform and personalization folder, and rename them

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8

FOLDER_NAME=mochi-high-five
FATHER_FOLDER_NAME=mochi-fox

# PROMPT_1="Turn the cat into a fox."
# SPECIAL_PROMPT_1="Turn the cat into a <kth> fox."

# PROMPT_2="Give the fox a pair of glasses."
# SPECIAL_PROMPT_2="Give the fox a pair of <kth> glasses."


PROMPT_2="Make the fox's body white."
SPECIAL_PROMPT_2="Make the fox's body <kth> white."



# FINE_TUNE_PERCENT=0.3
# LOSS_SELECTION=high_wight_constrastive
# EDIT_DIR="./data/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_1"
# DATA_DIR="/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}"
# IMAGE_DATA_DIR="/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}/rgb/1x"
# OUTPUT_DIR="./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_1"

# # Fine-tune the InstructPix2Pix
# python src/personalization.py \
#   --pretrained_model_name_or_path "timbrooks/instruct-pix2pix"  \
#   --enable_xformers_memory_efficient_attention \
#   --instance_data_dir "${EDIT_DIR}" \
#   --instance_prompt "${SPECIAL_PROMPT_1}" \
#   --class_data_dir "${OUTPUT_DIR}/class_samples" \
#   --class_prompt "${PROMPT_1}" \
#   --validation_prompt "${SPECIAL_PROMPT_1}" \
#   --output_dir "${OUTPUT_DIR}/personalization" \
#   --image_root="${IMAGE_DATA_DIR}" \
#   --validation_images 0_00000.png \
#     0_00050.png  \
#   --max_train_steps 1000 \
#   --checkpointing_steps 1000 \
#   --resolution 384 \
#   --vggt_feature_path "/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/vggt/${FOLDER_NAME}.pt" \
#   # --resume_from_checkpoint "${OUTPUT_DIR}/personalization/checkpoint-2000" \

# ## Load the pre-trained scene and edit the scene with the personalized InstructPix2Pix.
# python src/Deformable-3D-Gaussians/train.py \
#   -s "${DATA_DIR}" \
#   -m "${OUTPUT_DIR}" \
#   --iterations 30000 \
#   --edit_iterations 20000 \
#   --diffusion_model_checkpoint "${OUTPUT_DIR}/personalization/checkpoint-1000" \
#   --prompt "${SPECIAL_PROMPT_1}" \
#   --idx 42 \
#   --mask_iter 3000 \
#   --loss "${LOSS_SELECTION}" \
#   --fine_tune True \
#   --batch 1 \
#   --fine_tune_percent ${FINE_TUNE_PERCENT} \
#   --edited_image_train True \
#   --edited_image_train_percent 0.3

# ### Render the results
# python src/Deformable-3D-Gaussians/render.py -m "${OUTPUT_DIR}" --skip_test




# cp -r /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME} /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2


# rm -rf /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/rgb/1x/*

# cp ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_1/train/ours_50000/renders/* /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/rgb/1x/

# for file in /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/rgb/1x/*; do
#   mv "$file" "/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/rgb/1x/0_$(basename "$file")"
# done


mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/deform
mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/point_cloud
mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/personalization

cp -r ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_1/point_cloud/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/point_cloud/

mv ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/point_cloud/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/point_cloud/iteration_29000

cp -r ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_1/deform/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/deform/

mv ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/deform/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/deform/iteration_29000





FINE_TUNE_PERCENT=0.1
LOSS_SELECTION=high_wight_constrastive
EDIT_DIR=./data/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2
DATA_DIR=/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2
IMAGE_DATA_DIR=/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/rgb/1x
OUTPUT_DIR=./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2

# Fine-tune the InstructPix2Pix
python src/personalization.py \
  --pretrained_model_name_or_path "timbrooks/instruct-pix2pix"  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir "${EDIT_DIR}" \
  --instance_prompt "${SPECIAL_PROMPT_2}" \
  --class_data_dir "${OUTPUT_DIR}/class_samples" \
  --class_prompt "${PROMPT_2}" \
  --validation_prompt "${SPECIAL_PROMPT_2}" \
  --output_dir "${OUTPUT_DIR}/personalization" \
  --image_root="${IMAGE_DATA_DIR}" \
  --validation_images 0_00000.png \
    0_00050.png  \
  --max_train_steps 1000 \
  --checkpointing_steps 1000 \
  --resolution 384 \
  --vggt_feature_path "/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/vggt/${FOLDER_NAME}.pt" \

# Load the pre-trained scene and edit the scene with the personalized InstructPix2Pix.
python src/Deformable-3D-Gaussians/train.py \
  -s "${DATA_DIR}" \
  -m "${OUTPUT_DIR}" \
  --iterations 30000 \
  --load_checkpoint 29000 \
  --edit_iterations 20000 \
  --diffusion_model_checkpoint "${OUTPUT_DIR}/personalization/checkpoint-1000" \
  --prompt "${SPECIAL_PROMPT_2}" \
  --idx 42 \
  --mask_iter 3000 \
  --loss "${LOSS_SELECTION}" \
  --fine_tune True \
  --batch 1 \
  --fine_tune_percent ${FINE_TUNE_PERCENT} \
  --edited_image_train True \
  --edited_image_train_percent 0.1

### Render the results
python src/Deformable-3D-Gaussians/render.py -m "${OUTPUT_DIR}" --skip_test

# cp -r /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME} /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3

# rm -rf /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3/rgb/1x/*

# cp ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/train/ours_50000/renders/* /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3/rgb/1x/

# for file in /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3/rgb/1x/*; do
#   mv "$file" "/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3/rgb/1x/0_$(basename "$file")"
# done




# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# bash /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/CTRL-D_training/run_training_2.sh


# # print end time
# echo "All tasks completed successfully at $(date +'%Y-%m-%d %H:%M:%S')"

# for i in {4..14}
# do 
  # mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/deform
  # mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/point_cloud
  # mkdir -p ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/personalization

  # cp -r ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/point_cloud/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/point_cloud/

  # mv ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/point_cloud/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/point_cloud/iteration_29000

  # cp -r ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_2/deform/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/deform/

  # mv ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/deform/iteration_50000 ./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_${i}/deform/iteration_29000




   # 如果不需要mask用这条指令
#   FINE_TUNE_PERCENT=0.1
#   LOSS_SELECTION=high_wight_constrastive
#   EDIT_DIR=./data/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3
#   DATA_DIR=/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3
#   IMAGE_DATA_DIR=/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/0_Data/DyCheck/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_3/rgb/1x
#   OUTPUT_DIR=./output_edited_rended_image/${FATHER_FOLDER_NAME}/${FOLDER_NAME}_8


#   # Fine-tune the InstructPix2Pix
#   python src/personalization.py \
#     --pretrained_model_name_or_path "timbrooks/instruct-pix2pix"  \
#     --enable_xformers_memory_efficient_attention \
#     --instance_data_dir "${EDIT_DIR}" \
#     --instance_prompt "${SPECIAL_PROMPT_3}" \
#     --class_data_dir "${OUTPUT_DIR}/class_samples" \
#     --class_prompt "${PROMPT_3}" \
#     --validation_prompt "${SPECIAL_PROMPT_3}" \
#     --output_dir "${OUTPUT_DIR}/personalization" \
#     --image_root="${IMAGE_DATA_DIR}" \
#     --validation_images 0_00000.png \
#       0_00050.png  \
#     --max_train_steps 1000 \
#     --checkpointing_steps 1000 \
#     --resolution 384 \
#     --vggt_feature_path "/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/vggt/${FOLDER_NAME}.pt" \
#     # --resume_from_checkpoint "${OUTPUT_DIR}/personalization/checkpoint-2000" \

#   # Load the pre-trained scene and edit the scene with the personalized InstructPix2Pix.
#   python src/Deformable-3D-Gaussians/train.py \
#     -s "${DATA_DIR}" \
#     -m "${OUTPUT_DIR}" \
#     --iterations 30000 \
#     --load_checkpoint 29000 \
#     --edit_iterations 20000 \
#     --diffusion_model_checkpoint "${OUTPUT_DIR}/personalization/checkpoint-1000" \
#     --prompt "${SPECIAL_PROMPT_3}" \
#     --idx 42 \
#     --mask_iter 100 \
#     --loss "${LOSS_SELECTION}" \
#     --fine_tune True \
#     --batch 1 \
#     --fine_tune_percent ${FINE_TUNE_PERCENT} \
#     --edited_image_train True \
#     --edited_image_train_percent 0.1

#   ### Render the results
#   python src/Deformable-3D-Gaussians/render.py -m "${OUTPUT_DIR}" --skip_test


#   # print end time
#   echo "All tasks completed successfully at $(date +'%Y-%m-%d %H:%M:%S')"

# done

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
bash /mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/CTRL-D_training/run_training_2.sh