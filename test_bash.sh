# Testing example for Moving-MNIST
python test_whole.py \
--dataset 'satellite' --make_frame True \
--test_data_dir './test_samples' --test_result_dir './test_satellite' \
--checkpoint_load_file './pred_model.pt' \
--img_size 256 --img_channel 1 --memory_size 100 \
--short_len 8 --out_len 16 --long_len 16 \
--batch_size 1 \
--evaluate True
