# Evaluate different trained models
python sample_eval.py --use-cache --load-model-folder model_benchmark
python sample_eval.py --use-cache --load-model-folder model_benchmark_epoch200 --save-metrics
python sample_eval.py --use-cache --load-model-folder model_benchmark_epoch400 --save-metrics
python sample_eval.py --use-cache --load-model-folder model_benchmark_epoch600 --save-metrics
python sample_eval.py --use-cache --load-model-folder model_benchmark_epoch800 --save-metrics
python sample_eval.py --use-cache --load-model-folder model_benchmark_epoch1000 --save-metrics
python sample_eval.py --use-cache --load-model-folder model_pca --use-pca
python sample_eval.py --use-cache --load-model-folder model_savgol --use-savgol
python sample_eval.py --use-cache --load-model-folder model_wavelet --use-wavelet
python sample_eval.py --use-cache --load-model-folder model_savgol_pca --use-savgol --use-pca
python sample_eval.py --use-cache --load-model-folder model_wavelet_pca --use-wavelet --use-pca
python sample_eval.py --use-cache --load-model-folder model_circles1 --training-circles-num 1
python sample_eval.py --use-cache --load-model-folder model_circles2 --training-circles-num 2
python sample_eval.py --use-cache --load-model-folder model_circles3 --training-circles-num 3
python sample_eval.py --use-cache --load-model-folder model_circles4 --training-circles-num 4
python sample_eval.py --use-cache --load-model-folder model_one_forth --use-subset --subset-percentage 0.25
python sample_eval.py --use-cache --load-model-folder model_one_half --use-subset --subset-percentage 0.5
python sample_eval.py --use-cache --load-model-folder model_three_forth --use-subset --subset-percentage 0.75
python sample_eval.py --use-cache --load-model-folder model_skip_every_2 --skip-pins --skip-every 2
python sample_eval.py --use-cache --load-model-folder model_skip_every_4 --skip-pins --skip-every 4

# Evaluate fine-tuned models
python sample_eval.py --use-cache --load-model-folder model_circles1_finetuned2 --testing-circles-num 2
python sample_eval.py --use-cache --load-model-folder model_circles1_finetuned3 --testing-circles-num 3
python sample_eval.py --use-cache --load-model-folder model_circles1_finetuned4 --testing-circles-num 4
python sample_eval.py --use-cache --load-model-folder model_circles2_finetuned1 --testing-circles-num 1
python sample_eval.py --use-cache --load-model-folder model_circles2_finetuned3 --testing-circles-num 3
python sample_eval.py --use-cache --load-model-folder model_circles2_finetuned4 --testing-circles-num 4
python sample_eval.py --use-cache --load-model-folder model_circles3_finetuned1 --testing-circles-num 1
python sample_eval.py --use-cache --load-model-folder model_circles3_finetuned2 --testing-circles-num 2
python sample_eval.py --use-cache --load-model-folder model_circles3_finetuned4 --testing-circles-num 4
python sample_eval.py --use-cache --load-model-folder model_circles4_finetuned1 --testing-circles-num 1
python sample_eval.py --use-cache --load-model-folder model_circles4_finetuned2 --testing-circles-num 2
python sample_eval.py --use-cache --load-model-folder model_circles4_finetuned3 --testing-circles-num 3