#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:04:06 2022

@author: ghiggi
"""
##----------------------------------------------------------------------------.
import numpy as np
from xforecasting.utils.ar import (
    check_ar_settings,
    plot_ar_settings,
    get_first_valid_idx,
    get_last_valid_idx,
    get_dict_Y,
    get_dict_X_dynamic,
    get_dict_X_bc,
    get_dict_stack_info,
)

##----------------------------------------------------------------------------.
### Example - Classical Autoregressive
input_k = np.array([-9, -6, -3])
output_k = np.array([0])
ar_iterations = 6
forecast_cycle = 3

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
)

print(get_first_valid_idx(input_k))
print(
    get_last_valid_idx(
        output_k=output_k, forecast_cycle=forecast_cycle, ar_iterations=ar_iterations
    )
)
### Example DIRECT OUTPUT
forecast_cycle = 1
ar_iterations = 0
input_k = np.array([-3, -2, -1])
output_k = np.array([0, 1, 2, 3, 4, 5, 6, 7])

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
)

print(get_first_valid_idx(input_k))
print(
    get_last_valid_idx(
        output_k=output_k, forecast_cycle=forecast_cycle, ar_iterations=ar_iterations
    )
)

### Example AR OUTPUT
forecast_cycle = 1
ar_iterations = 7
input_k = np.array([-3, -2, -1])
output_k = np.array([0])

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
)


##----------------------------------------------------------------------------.
### Example - Fixed DELTA T
delta_t = 6
n_input_lags = 3
n_output_timesteps = 1
forecast_cycle = delta_t
ar_iterations = 4
input_k = np.cumsum(-1 * np.repeat(delta_t, repeats=n_input_lags))[::-1]
output_k = np.cumsum(
    np.concatenate((np.array([0]), np.repeat(delta_t, repeats=n_output_timesteps - 1)))
)

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
)

print(get_first_valid_idx(input_k))
print(
    get_last_valid_idx(
        output_k=output_k, forecast_cycle=forecast_cycle, ar_iterations=ar_iterations
    )
)

##----------------------------------------------------------------------------.
### Example - Multi-Temporal Output
input_k = np.array([-7, -5, -3, -1])
output_k = np.array([0, 1, 3, 7])
ar_iterations = 6
forecast_cycle = 2

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=False,
)

##----------------------------------------------------------------------------.
### Example - Chess pattern
input_k = np.array([-8, -6, -4, -2])
output_k = np.array([0, 2, 4, 7])
ar_iterations = 6
forecast_cycle = 1

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(
    input_k, output_k, forecast_cycle, ar_iterations, stack_most_recent_prediction=True
)
plot_ar_settings(
    input_k, output_k, forecast_cycle, ar_iterations, stack_most_recent_prediction=False
)

# -----------------------------------------------------------------------------.
### Example dictionary for data retrieval and stacking
ar_iterations = 5
forecast_cycle = 3
output_k = np.array([0, 3])
input_k = np.array([-9, -6, -3])

plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=False,
)

plot_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)

dict_Y = get_dict_Y(
    ar_iterations=ar_iterations, forecast_cycle=forecast_cycle, output_k=output_k
)
dict_X_dynamic = get_dict_X_dynamic(
    ar_iterations=ar_iterations, forecast_cycle=forecast_cycle, input_k=input_k
)
dict_X_bc = get_dict_X_bc(
    ar_iterations=ar_iterations, forecast_cycle=forecast_cycle, input_k=input_k
)
dict_Y_to_stack, dict_Y_to_remove = get_dict_stack_info(
    ar_iterations=ar_iterations,
    forecast_cycle=forecast_cycle,
    input_k=input_k,
    output_k=output_k,
    stack_most_recent_prediction=False,
)

# ----------------------------------------------------------------------------.
### Check validity of something do not work
input_k = np.array([-7, -5, -3, -1])
output_k = np.array([0, 1, 3, 7])
ar_iterations = 2
forecast_cycle = 5

check_ar_settings(
    input_k=input_k,
    output_k=output_k,
    forecast_cycle=forecast_cycle,
    ar_iterations=ar_iterations,
    stack_most_recent_prediction=True,
)
plot_ar_settings(input_k, output_k, forecast_cycle, ar_iterations, hatch=True)

# - Use Hatch=False allow to still visualize the AR setting
plot_ar_settings(input_k, output_k, forecast_cycle, ar_iterations, hatch=False)

# -----------------------------------------------------------------------------.
