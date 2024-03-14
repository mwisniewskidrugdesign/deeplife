padding = 0
conv1_out_channels = 16
filter_size = 3
import math

linear_input = (conv1_out_channels * 2) * ((math.floor(
    (math.floor(110 + (2 * padding) - filter_size) + 1) + (2 * padding) - filter_size) + 1) / 2) ** 2

a=1
