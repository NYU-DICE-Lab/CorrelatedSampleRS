"""
Generate result files for latex files
"""

import os
from matplotlib import pyplot as plt
from smoothadv.analyze import Line, ApproximateAccuracy, plot_certified_accuracy
import glob
import numpy as np
from pathlib import Path 


def get_accuracy_plots(inpath, legend, outpath, max_radius, radius_step=0.01):
    """
    Generate accuracy plots for given legends
    """
    radius_start=0.0
    appx_acc = ApproximateAccuracy(inpath)
    radii = np.arange(radius_start, max_radius + radius_step, radius_step)
    method = Line(appx_acc)
    accuracies = method.quantity.at_radii(radii)
    print(outpath)
    with open(outpath, 'w') as f:
        f.write('radius,accuracy\n')
        for rad, acc in zip(radii, accuracies):
            f.write(f'{rad:.03f},{acc}\n')    
    return accuracies, radii

if not os.path.exists('paper_results'):
    os.makedirs('paper_results')
rpath = Path('paper_results')

## CIFAR results

if not os.path.exists(rpath / 'CIFAR10'):
    os.makedirs(rpath / 'CIFAR10')

cifar_path = Path(rpath / 'CIFAR10')

# Max, smoothmax and salman

# sigma = 0.25
# max_radius=1.0
# radius_step=0.01
# get_accuracy_plots(f'certify_results_salman_new_nopatch_3232/output_resnet110_{sigma}_32_1_mean.csv' , '', cifar_path / f'salman_base_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth/output_resnet110_{sigma}_32_2_max.csv' , '', cifar_path / f'max_1_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_100/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_100_{sigma*100}.csv', max_radius, radius_step)


# sigma = 0.5
# max_radius=2.0
# radius_step=0.01
# get_accuracy_plots(f'certify_results_salman_new_nopatch_3232/output_resnet110_{sigma}_32_1_mean.csv' , '', cifar_path / f'salman_base_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth/output_resnet110_{sigma}_32_2_max.csv' , '', cifar_path / f'max_1_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_100/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_100_{sigma*100}.csv', max_radius, radius_step)

# sigma = 1.0
# max_radius=4.0
# radius_step=0.01
# get_accuracy_plots(f'certify_results_salman_new_nopatch_3232/output_resnet110_{sigma}_32_1_mean.csv' , '', cifar_path / f'salman_base_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth/output_resnet110_{sigma}_32_2_max.csv' , '', cifar_path / f'max_1_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_100/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_100_{sigma*100}.csv', max_radius, radius_step)

# ## Smoothmax v/s salman (number of patches)

# sigma=0.25
# max_radius=1.0
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_50/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_50_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_25/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_25_{sigma*100}.csv', max_radius, radius_step)

# sigma=0.5
# max_radius=2.0
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_50/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_50_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_25/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_25_{sigma*100}.csv', max_radius, radius_step)

# sigma=1.0
# max_radius=4.0
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_50/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_50_{sigma*100}.csv', max_radius, radius_step)
# get_accuracy_plots(f'certify_results_salman_patchsmooth_smoothmax_randompatches_25/output_resnet110_{sigma}_32_1_max.csv' , '', cifar_path / f'smoothmax_25_{sigma*100}.csv', max_radius, radius_step)

sigma=1.0
max_radius=2.0
radius_step=0.01
get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_0.25_16_16_max.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_0.25_16_16_max", max_radius, radius_step)
get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_0.5_16_16_max.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_0.5_16_16_max", max_radius, radius_step)
get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_1.0_16_16_max.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_max/output_resnext-101_1.0_16_16_max", max_radius, radius_step)

get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_0.25_16_16_baseline.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_0.25_16_16_baseline", max_radius, radius_step)
get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_0.5_16_16_baseline.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_0.5_16_16_baseline", max_radius, radius_step)
get_accuracy_plots(f'/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_1.0_16_16_baseline.csv' , '', "/home/mp5847/src/CorrelatedSampleRS/vidtest_baseline/output_resnext-101_1.0_16_16_baseline", max_radius, radius_step)