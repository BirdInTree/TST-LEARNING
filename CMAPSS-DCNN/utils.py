from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import torch

# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(
        max_cycle.to_frame(name="max_cycle"), left_on="unit_nr", right_index=True
    )

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond["setting_1"] = abs(df_op_cond["setting_1"].round())
    df_op_cond["setting_2"] = abs(df_op_cond["setting_2"].round(decimals=2))

    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond["op_cond"] = (
        df_op_cond["setting_1"].astype(str)
        + "_"
        + df_op_cond["setting_2"].astype(str)
        + "_"
        + df_op_cond["setting_3"].astype(str)
    )

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names, scale_type: str = "std-mean"):
    # 对每个操作条件分别进行标准化
    if scale_type == "std-mean":
        for condition in df_train["op_cond"].unique():
            # 1. 选择当前操作条件下的训练数据和测试数据
            train_condition_data = df_train.loc[
                df_train["op_cond"] == condition, sensor_names
            ]
            test_condition_data = df_test.loc[
                df_test["op_cond"] == condition, sensor_names
            ]

            # 2. 计算训练集的均值和标准差（只在训练集上计算）
            mean = train_condition_data.mean()
            std = train_condition_data.std()

            # 3. 防止除以零的情况，如果标准差为0，则设为1（因为该特征的值都是常数）
            std.replace(0, 1.0, inplace=True)

            # 4. 对训练集进行标准化
            df_train.loc[df_train["op_cond"] == condition, sensor_names] = (
                train_condition_data - mean
            ) / std

            # 5. 使用训练集的均值和标准差，对测试集进行标准化
            df_test.loc[df_test["op_cond"] == condition, sensor_names] = (
                test_condition_data - mean
            ) / std
    elif scale_type == "max-min":
        for condition in df_train["op_cond"].unique():
            # 1. 选择当前操作条件下的训练数据和测试数据
            train_condition_data = df_train.loc[
                df_train["op_cond"] == condition, sensors
            ]
            test_condition_data = df_test.loc[df_test["op_cond"] == condition, sensors]

            # 2. 计算训练集的最大值和最小值（只在训练集上计算）
            max_value = train_condition_data.max()
            min_value = train_condition_data.min()

            # 3. 对训练集进行最大最小化
            df_train.loc[df_train["op_cond"] == condition, sensors] = (
                train_condition_data - min_value
            ) / (max_value - min_value)

            # 4. 使用训练集的最大值和最小值，对测试集进行最大最小化
            df_test.loc[df_test["op_cond"] == condition, sensors] = (
                test_condition_data - min_value
            ) / (max_value - min_value)
    else:
        pass
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    df[sensors] = (
        df.groupby("unit_nr")[sensors]
        .apply(lambda x: x.ewm(alpha=alpha).mean())
        .reset_index(level=0, drop=True)
    )

    # second, drop first n_samples of each unit_nr to reduce filter delay
    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = (
        df.groupby("unit_nr")["unit_nr"]
        .transform(create_mask, samples=n_samples)
        .astype(bool)
    )
    df = df[mask]

    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]

    # -1 and +1 because of Python indexing
    for start, stop in zip(
        range(0, num_elements - (sequence_length - 1)),
        range(sequence_length, num_elements + 1),
    ):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df["unit_nr"].unique()

    data_gen = (
        list(gen_train_data(df[df["unit_nr"] == unit_nr], sequence_length, columns))
        for unit_nr in unit_nrs
    )
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]

    # -1 because I want to predict the rul of that last row in the sequence, not the next row
    return data_matrix[sequence_length - 1 : num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df["unit_nr"].unique()

    label_gen = [
        gen_labels(df[df["unit_nr"] == unit_nr], sequence_length, label)
        for unit_nr in unit_nrs
    ]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(
            shape=(sequence_length, len(columns)), fill_value=mask_value
        )  # pad
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values  # fill with available data
    else:
        data_matrix = df[columns].values

    # specifically yield the last possible sequence
    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def get_data(
    dataset: str = "FD001",
    sensors: list = ["s_2", "s_3"],
    sequence_length: int = 30,
    alpha: Optional[float] = None,
    threshold: int = 125,
    scale_type: Optional[Literal["max-min", "std-mean", None]] = None,
    random_state: int = 42,
    train_size: float = 0.8,
    plot_unit: list[np.array] = np.array([1,]),
):
    # files
    dir_path = "./data/"
    train_file = "train_" + dataset + ".txt"
    test_file = "test_" + dataset + ".txt"
    # columns
    index_names = ["unit_nr", "time_cycles"]
    setting_names = ["setting_1", "setting_2", "setting_3"]
    sensor_names = ["s_{}".format(i + 1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
    # data readout
    train = pd.read_csv(
        (dir_path + train_file),
        sep=r"\s+",
        header=None,
        dtype=np.float32,
        names=col_names,
    )
    test = pd.read_csv(
        (dir_path + test_file),
        sep=r"\s+",
        header=None,
        dtype=np.float32,
        names=col_names,
    )
    y_test = pd.read_csv(
        (dir_path + "RUL_" + dataset + ".txt"),
        sep=r"\s+",
        header=None,
        names=["RemainingUsefulLife"],
    )

    # create RUL values according to the piece-wise target function
    train = add_remaining_useful_life(train)
    # train['RUL'].clip(upper=threshold, inplace=True)
    train.loc[:, "RUL"] = train["RUL"].clip(upper=threshold)

    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]
    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(
        X_train_pre, X_test_pre, sensors, scale_type
    )
    if alpha is not None:
    #exponential smoothing
        X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
        X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)
    else:
        pass

    # train-val split
    gss = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    # generate the train/val for *each* sample -> for that we iterate over the train and val units we want
    # this is a for that iterates only once and in that iterations at the same time iterates over all the values we want,
    # i.e. train_unit and val_unit are not a single value but a set of training/vali units
    for train_unit, val_unit in gss.split(
        X_train_pre["unit_nr"].unique(), groups=X_train_pre["unit_nr"].unique()
    ):
        train_unit = X_train_pre["unit_nr"].unique()[train_unit]  # gss returns indexes and index starts at 1
        val_unit = X_train_pre["unit_nr"].unique()[val_unit]
        print("train_unit:", train_unit, "val_unit:", val_unit)
        x_train = gen_data_wrapper(X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(X_train_pre, sequence_length, ["RUL"], train_unit)

        x_val = gen_data_wrapper(X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(X_train_pre, sequence_length, ["RUL"], val_unit)

    #生成用于一步预测绘图的train数据
    x_train_plot = gen_data_wrapper(X_train_pre, sequence_length, sensors, plot_unit)
    y_train_plot = gen_label_wrapper(X_train_pre, sequence_length, ["RUL"], plot_unit)
    # x_test_plot = gen_data_wrapper(X_test_pre, sequence_length, sensors, plot_unit)
    # y_test_plot = gen_label_wrapper(X_test_pre, sequence_length, ["RUL"], plot_unit)
    # create sequences for test
    test_gen = (
        list(
            gen_test_data(
                X_test_pre[X_test_pre["unit_nr"] == unit_nr],
                sequence_length,
                sensors,
                -99.0,
            )
        )
        for unit_nr in X_test_pre["unit_nr"].unique()
    )
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test["RemainingUsefulLife"], x_train_plot, y_train_plot


# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    sensors = ["s_3", "s_4", "s_7", "s_11", "s_12"]
    squence_length = 30
    alpha = 0.1
    threshold = 125
    x_train, y_train, x_val, y_val, x_test, y_test, x_test_plot, y_test_plot = get_data(
        "FD001", sensors, squence_length, alpha, threshold, plot_unit=np.array([2])
    )
    #(258, 30, 5) (258, 1)
    #绘制训练数据的RUL曲线 y_train_plot
    print(y_test_plot)
    x = np.arange(1,y_test_plot.shape[0]+1)
    plt.plot(x, y_test_plot[:,0], label="train_rul")
    plt.show()
