# Training from Scratch with different configurations
python sample_train.py --use-cache -t -e -s --save-model-folder model_benchmark -m --save-every 200 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_downscale_32 -d --downscale-resolution 32 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_downscale_64 -d --downscale-resolution 64 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_downscale_96 -d --downscale-resolution 96 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_pca --use-pca --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_savgol --use-savgol --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_wavelet --use-wavelet --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_savgol_pca --use-savgol --use-pca --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_wavelet_pca --use-wavelet --use-pca --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_circles1 --training-circles-num 1 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_circles2 --training-circles-num 2 --use-scheduler --decay-rate 0.7 --decay-epochs 150 
python sample_train.py --use-cache -t -e -s --save-model-folder model_circles3 --training-circles-num 3 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_circles4 --training-circles-num 4 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_one_forth --use-subset --subset-percentage 0.25 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_one_half --use-subset --subset-percentage 0.5 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_three_forth --use-subset --subset-percentage 0.75 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_skip_every_2 --skip-pins --skip-every 2 --use-scheduler --decay-rate 0.7 --decay-epochs 150
python sample_train.py --use-cache -t -e -s --save-model-folder model_skip_every_4 --skip-pins --skip-every 4 --use-scheduler --decay-rate 0.7 --decay-epochs 150

# Fine-tuning existing models with different number of circles
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles1 -s --save-model-folder model_circles1_finetuned2 --training-circles-num 2 --testing-circles-num 2 --epochs 250 --learning-rate 0.0001 
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles1 -s --save-model-folder model_circles1_finetuned3 --training-circles-num 3 --testing-circles-num 3 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles1 -s --save-model-folder model_circles1_finetuned4 --training-circles-num 4 --testing-circles-num 4 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles2 -s --save-model-folder model_circles2_finetuned1 --training-circles-num 1 --testing-circles-num 1 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles2 -s --save-model-folder model_circles2_finetuned3 --training-circles-num 3 --testing-circles-num 3 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles2 -s --save-model-folder model_circles2_finetuned4 --training-circles-num 4 --testing-circles-num 4 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles3 -s --save-model-folder model_circles3_finetuned1 --training-circles-num 1 --testing-circles-num 1 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles3 -s --save-model-folder model_circles3_finetuned2 --training-circles-num 2 --testing-circles-num 2 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles3 -s --save-model-folder model_circles3_finetuned4 --training-circles-num 4 --testing-circles-num 4 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles4 -s --save-model-folder model_circles4_finetuned1 --training-circles-num 1 --testing-circles-num 1 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles4 -s --save-model-folder model_circles4_finetuned2 --training-circles-num 2 --testing-circles-num 2 --epochs 250 --learning-rate 0.0001
python sample_train.py --use-cache -t -e -l --load-model-folder model_circles4 -s --save-model-folder model_circles4_finetuned3 --training-circles-num 3 --testing-circles-num 3 --epochs 250 --learning-rate 0.0001