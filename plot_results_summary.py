import matplotlib.pyplot as plt

import numpy as np

models = ["ConvNext\nSmall\nPretrained","ConvNext\nSmall\nScratch","ConvNext\nBase\nPretrained","ConvNext\nBase\nScratch",
          "Res50\nPretrained","Res50\nScratch","Res101\nPretrained","Res101\nScratch"]

models2 = ["scratch\nw/o_aug","scratch\nw_aug","scratch_IM\nstarting\nweights\nw/o_aug",
           "scratch_IM\nstarting\nweights\nw/o_aug",
           "TU_SC\nw/o_aug","TU_SC\nw_aug","TU_IM\nw/o_aug","TU_IM\nw_aug"]

if __name__ == "__main__":
    #2
    # validation_scores = np.array([81.54, 65.2, 82.3, 65.09, 77.11, 68.81, 79.21, 70.16])
    # validation_stds = np.array([0.67, 0.72, 0.45, 1.25, 0.01, 0.94, 0.52, 1.69])
    #
    # test_scores = np.array([81.49, 64.2, 81.71, 64.22, 76.96, 68.62, 79.4, 69.67])
    # test_stds =   np.array([0.5, 0.28, 0.26, 0.83, 0.61, 0.69, 0.64, 1.65])
    #
    # 1
    # validation_scores = np.array([81.71, 65.13, 81.71, 65.32, 76.93, 71.65, 78.98, 71.3])
    # validation_stds = np.array([1, 1.37, 0.5, 0.32, 0.77, 1.6, 1.52, 0.89])
    #
    # test_scores = np.array([81.97, 64.68, 81.86, 64.76, 77.03, 71.88, 79.12, 71.7])
    # test_stds =   np.array([0.81, 0.53, 0.34, 0.8, 0.2, 1.18, 1.07, 1.41])
    #
    validation_scores = [2.89,3.03,2.99,3.01,2.75,2.84,2.75,2.81]
    validation_stds = [0.08,0.14,0.11,0.13,0.15,0.17,0.11,0.11]
    test_scores = [2.88,3.05,3,3.03,2.87,2.82,2.9,2.89]
    test_stds = [0.09,0.08,0.08,0.1,0.07,0.02,0.16,0.12]
    plt.figure(figsize=(20, 15))
    #
    x = np.arange(8)
    width = 0.4
    colors = plt.cm.rainbow(np.linspace(0, 1, 2))
    plt.bar(x-0.2, validation_scores, width , color=colors[0])
    plt.bar(x+0.2, test_scores, width, color=colors[1])
    plt.errorbar(x-0.2, validation_scores, capsize=10, capthick=5,
                              ecolor='black',yerr=validation_stds, elinewidth=5,fmt='none')
    plt.errorbar(x+0.2, test_scores, capsize=10, capthick=5,
                              ecolor='black',yerr=test_stds, elinewidth=5,fmt='none')
    plt.xticks(x, models2,fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(bottom=2)
    plt.legend(['Validation accuracy','Test accuracy'],
               fontsize = 20)
    plt.title("ConvNext Base : Collected Alzheimer regression results",
              color='navy',
              fontsize=30,
              fontweight='bold')
    plt.savefig('scores3.png')


