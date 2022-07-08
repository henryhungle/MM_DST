
model_path=exps/mmdst_dvd_vdtn/ep1_08-07-22_4.78849.pth
python main.py inference_dst --inference_config configs/inference_vdtn_config.json --model_path $model_path --inference_style greedy

