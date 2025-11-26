# Run reconstructions for various trained models
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch200 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch400 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch600 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch800 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_benchmark_epoch1000 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_32 -d --downscale-resolution 32 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_64 -d --downscale-resolution 64 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_downscale_96 -d --downscale-resolution 96 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_pca --use-pca --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_savgol_pca --use-savgol --use-pca --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_wavelet_pca --use-wavelet --use-pca --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_circles1 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_circles2 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_circles3 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_circles4 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_one_forth --use-subset --subset-percentage 0.25 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_one_half --use-subset --subset-percentage 0.5 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_three_forth --use-subset --subset-percentage 0.75 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_skip_every_2 --skip-pins --skip-every 2 --sample-id 42
python sample_reconstruction.py --use-cache --load-model-folder model_skip_every_4 --skip-pins --skip-every 4 --sample-id 42