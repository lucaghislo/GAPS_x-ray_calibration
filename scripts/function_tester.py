from calculate_xray_gain import *

filepath_fdt_data = r"input\transfer_function_data\L4R0M0_TransferFunction.dat"

gain, pedestal = get_linear_gain(filepath_fdt_data, 5, 5, 100)

print(gain)
print(pedestal)
