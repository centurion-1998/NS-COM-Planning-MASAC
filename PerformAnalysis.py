import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

class PerfAnalysis:
    def __init__(self, fig_path, perf_data_path, data_paths_input, label_list, data_config, unit):
        """
        初始化性能分析器
        :param fig_path: 保存图像的路径
        :param perf_data_path: 包含实验数据的根路径
        :param data_paths_input: 包含各组实验数据路径的列表
        :param label_list: 每组实验的标签列表
        :param data_config: 指标名称列表
        :param unit: 每个指标的单位列表
        """
        self.fig_path = fig_path
        self.perf_data_path = perf_data_path
        self.data_config = data_config
        self.data = []
        self.data_path_list = []
        self.label_name_list = label_list
        self.unit = unit

        # 确保保存路径存在
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        # 加载所有实验的数据
        for data_path_origin in data_paths_input:
            data_path = os.path.join(perf_data_path, data_path_origin)
            print("DATA_PATH:   ", data_path)
            self.data.append(np.load(data_path))
            self.data_path_list.append(data_path)

    def plot_comparison(self):
        """
        对比不同组实验的同一指标
        """
        # 遍历每个指标
        for i, data_name in enumerate(self.data_config):
            plt.figure(figsize=(10, 6))

            data_unit = self.unit[i]

            # 提取所有实验的当前指标数据
            all_mean_values = []
            all_std_values = []

            for j, path in enumerate(self.data_path_list):
                label_name = self.label_name_list[j]

                data = self.data[j][:, :, i]
                if i == 2:
                    data /= 6
                    plt.ylim([0.6,1])

                mean_values = np.nanmean(data, axis=0)
                std_values = np.nanstd(data, axis=0)  # 计算标准差

                all_mean_values.append(mean_values)
                all_std_values.append(std_values)

            # 计算所有曲线的最大值，确定统一的数量级
            all_max_values = np.concatenate([mean + std for mean, std in zip(all_mean_values, all_std_values)])
            max_value = np.nanmax(all_max_values)
            exponent = int(np.floor(np.log10(max_value)))
            scale = 10 ** exponent

            # 绘制所有曲线
            for j, (mean_values, std_values) in enumerate(zip(all_mean_values, all_std_values)):
                label_name = self.label_name_list[j]

                # 绘制均值曲线
                line, = plt.plot(mean_values / scale, label=label_name, linewidth=2)
                line_color = line.get_color()  # 获取当前曲线的颜色

                # 填充均值上下一个标准差的区域
                plt.fill_between(range(mean_values.shape[0]),
                                 (mean_values - std_values) / scale,
                                 (mean_values + std_values) / scale,
                                 color=line_color, alpha=0.2)

            # 添加标题和标签
            plt.title(f'Comparison of {data_name} over Steps', fontsize=20)
            plt.xlabel('Steps', fontsize=16)
            plt.ylabel(f'{data_name} (×10$^{{{exponent}}}$ {data_unit})', fontsize=16)
            plt.legend(fontsize=14)
            plt.grid(True)

            # 纵坐标仅保留有效数字部分
            def sci_formatter(x, pos):
                return f'{x:.2f}'.rstrip('0').rstrip('.')

            plt.gca().yaxis.set_major_formatter(FuncFormatter(sci_formatter))

            # 调整横纵坐标数字字体大小
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_path, f"{data_name}_comparison.png"))
            plt.close()

if __name__ == "__main__":
    # 假设有两组实验数据
    data_config = ["Min Fluent", "Sum Fluent", "JFI", "Energy"]
    fig_path = "./data/perf_log/"
    perf_data_path = "./data/perf_log/perf_data/"

    unit = ["Bit/s·Hz", "Bit/s·Hz", "", "kW"]

    # 创建性能分析器
    perf_analysis = PerfAnalysis(fig_path, perf_data_path,
                                ['Base.npy', 'GAT_FREE.npy', 'GRU_FREE.npy'],
                                ['Proposed Algorithm', 'GAT-Free', 'GRU-Free'],
                                data_config, unit)

    # 绘制对比图
    perf_analysis.plot_comparison()