import numpy as np
import matplotlib.pyplot as plt
from thesis import BaseGenerator, BinarySegmentationModel, HMMModel, BinaryHMM

if __name__ != "__main__":
    raise NotImplementedError

bg = BaseGenerator([0.3, 0.01, 0.5], [500, 1000, 400])
x = bg.generate(5000)
print x

# bsm = BinarySegmentationModel(0.05)
# hmm = HMMModel(BinaryHMM(11))
# hmm.plot_prediction(x, bg, show=False)
for threshold in [0.1, 0.05, 0.02]:
    bsm = BinarySegmentationModel(threshold)
    bsm.plot_prediction(x, bg, show=False)

bsm = BinarySegmentationModel(0.01)
bsm.plot_prediction(x, bg)

# z = bsm.change_points_to_sequence([0] + list(np.cumsum(bg.durations[:-1])) + [len(x)], bg.probabilities)
# y = bsm.segmentation(x)
# plt.plot(z)
# plt.plot(y)
# plt.show()

# z = []
# z_theory = []
# for t in range(2*bsm.window+1, len(x)):
#     max_z = 0
#     for k in range(bsm.window, t - bsm.window):
#         n1 = k
#         n2 = t - k
#         p1 = np.mean(x[:k])
#         p2 = np.mean(x[k:t])
#         max_z = max((max_z, bsm.test_statistic(n1, n2, p1, p2)))
#     z.append(max_z)
#     z_theory.append(bsm.gauss_threshold(np.mean(x[:t]), t))
#
# ind = np.arange(len(x))
# plt.plot(ind, x, 'b+')
# plt.plot(z)
# plt.plot(z_theory)
# plt.show()


#     n_tot = 500
#     # l_increase = [i for i in range(n_tot+1) if np.random.rand() < float(i)/n_tot]
#     # print l_increase
#     # faults = one_hot(WeibullGenerator(a=5., lambd=1).run(n_tot), n_tot)
#
#     raise Exception()
#     # faults = one_hot(
#     #     [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
#     #      33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 58, 61, 62, 63, 64,
#     #      65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97,
#     #      103, 104, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 130, 131, 133, 134, 135, 136, 137, 138,
#     #      139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
#     #      161, 162, 163, 195, 196, 197, 201, 202, 203, 204, 205, 206, 207, 213, 214, 215, 217, 218, 220, 221, 222, 223,
#     #      225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
#     #      247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
#     #      269, 270, 272, 273, 274, 275, 276, 277, 283, 289, 292, 293, 294, 295, 297, 302, 306, 307, 308, 309, 310, 311,
#     #      315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338,
#     #      339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 353, 354, 355, 356, 357, 358, 359, 360, 361,
#     #      362, 363, 364, 365, 366, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385,
#     #      386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
#     #      408, 409, 410, 411, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 432, 433,
#     #      434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
#     #      456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476], 476)
#
#     rolling = moving_average(faults)
#     # exp1 = numpy_ewma_vectorized(faults)
#     # exp2 = numpy_ewma_vectorized(rolling)
#     # diff1 = differential(exp1)
#     # diff2 = differential(exp2, 10)
#     l = len(faults)
#     plt.plot(faults, '.')
#     # plt.plot(base)
#     # plt.plot(rolling)
#     # plt.plot([np.std(rolling) for i in range(len(rolling))])
#     # plt.plot([split_std(rolling, i) for i in range(len(rolling))])
#     # plt.plot([split_mean(rolling, i) for i in range(len(rolling))])
#     # plt.plot(exp1)
#     # plt.plot(exp2)
#     # plt.plot(diff1)
#     # plt.plot(diff2)
#
#     score_, seg_ = segmentation(faults, 20)
#     print(score_)
#     seg_ = [0] + seg_ + [len(faults)]
#     scores_ = []
#     for i in range(len(seg_)-1):
#         mean_value = np.mean(faults[seg_[i]: seg_[i+1]])
#         scores_ += [mean_value, mean_value]
#     l_ = [[i, i+1] for i in seg_]
#     plt.plot([item for sublist in l_ for item in sublist][1:-1], scores_)
#     for s in seg_:
#         plt.plot([s, s], [0, 1], 'k')
#
#     plt.show()

