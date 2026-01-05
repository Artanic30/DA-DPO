source activate bpo_of
cd /public/home/qiult/projects/BPO-main


base_model=/inspurfs/group/hexm/VL_data/pretrained_models/llava-v1.5-13b

exp_name=13B_DADPO_repo
mkdir -p output/"$exp_name"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port 23101 llava/train/da_dpo.py \
    --mm_projector_lr 2e-6 \
    --mm_projector_type mlp2x_gelu \
    --learning_rate 2e-6 \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 256 \
    --model_name_or_path ${base_model} \
    --version v1 \
    --data_path data/BPO/bpo_instruct_qa_pt_clip_mixed_scored.json \
    --image_folder ./ \
    --vision_tower /inspurfs/group/hexm/VL_data/pretrained_models/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./output/${exp_name} \
    --num_train_epochs 1 \
    --subset_percent -1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --lazy_preprocess True \
    --beta 0.1 \
    2>&1 | tee -a output/${exp_name}/output.log
