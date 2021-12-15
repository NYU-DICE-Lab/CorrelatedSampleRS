# CorrelatedSampleRS

[Salman et al.](https://arxiv.org/abs/1906.04584) propose a randomized smoothing model to certify the performance of input images. We propose a (almost free) modified smooth classifier, that instead generates certificates on overlapping patches. The new smooth classifier improves over Salman et al. by ~10%, while also reducing the waterfall effects of adding noise. We provide theoretical and empirical evidence of the improvement.

We improve certificates for existing smooth classifiers. 

To replicate results certificates with our modified patch-based classifier;

```
    python3 infer_certify_pretrained_salman_uncorrelated.py cifar10 -dpath <path-to-dataset> -mp <path-to-saved-model> -mt <model-type> --patch -ni 100 -ps 32 -pstr 4 -sigma 0.25 --N0 100 --N 10000 -o <path-to-output-directory> --batch 400 -rm <max/min/mean> -ns 36 
```

To replicate results from Salman et al.

```
    python3  infer_certify_pretrained_salman_uncorrelated.py cifar10 -dpath <path-to-dataset> -mp <path-to-saved-model> -mt <model-type> -ni 100 -sigma 0.25 --N0 100 --N 10000 -o <path-to-output-directory> --batch 400 -rm <max/min/mean> -ns 32 
```

Results currently are posted at: https://www.overleaf.com/read/snmrwpgrpwrh

