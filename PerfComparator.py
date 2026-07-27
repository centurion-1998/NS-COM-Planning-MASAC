#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PerfComparator.py - 性能指标对比工具（支持滑动窗口平滑、异常剔除、生还率独立绘图）

用途：读取由 GraphTester.py 生成的 *_tester_results_data.npy 文件，
      对多个算法的性能指标进行可视化对比，并输出统计 CSV。

文件命名规则：
    支持在算法名前添加数字前缀和下划线来控制曲线绘制顺序。
    例如：0_Proposed_para_tester_results_data.npy 表示算法 "Proposed" 排第0位。
    不带数字前缀的算法将排在最后，按字母顺序排列。

线型区分（不含 ':'）：
    主曲线和生还率曲线按排序顺序依次使用线型：实线、虚线、点划线、自定义虚线。
    线型列表：['-', '--', '-.', (0, (5,2,1,2))]  # 最后一种为长虚线加短点

数据维度约定：
    npy 数组形状为 (num_episodes, max_steps, num_metrics)
    其中 num_metrics = 7，顺序固定为：
        [Min Fluent, Sum Fluent, JFI, Energy, Delay, Min Eff, Total Eff]

配置说明：
    在 __main__ 中的 metrics_config 字典中定义每个指标的绘图参数。
    支持以下字段：
        - display:         图例和纵轴显示名称
        - unit:            单位（将显示在纵轴）
        - ylim:            纵轴范围 (min, max)，None 表示自动
        - force_natural:   True 时强制使用普通数值显示（不缩放），
                           False 时根据数据自动选择科学计数法（×10^n）
        - scale:           数据缩放因子，默认为 1.0。所有数据将乘以该值后再处理。
        - smooth_window:   滑动窗口大小（步数），默认为 None 或 0 表示不平滑。
                           建议值 5~20，取决于数据步数。
        - outlier_threshold: 若指定（如 5.0），则基于平滑后的数据剔除 |x-mean| > threshold*std 的点。
        - outlier_mode:    'step' 或 'global'（默认 'step'），
                           'step' 按每个 step 独立计算，'global' 将所有数据合并计算。

使用方法：
    1. 确保 perf_log/ 目录下存在各算法的 *_tester_results_data.npy 文件。
    2. 按需修改 metrics_config 配置。
    3. 运行本脚本，将在 ./output/ 目录下生成各指标的对比图 (PNG) 和生还率图。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
from matplotlib.ticker import FuncFormatter

# 线型列表（不含 ':'，更清晰）
LINESTYLES = ['-']  # 实线、虚线、点划线、长虚线+短点


class PerfComparator:
    """
    性能对比类：加载数据、平滑、剔除异常、绘图、导出统计。
    """

    def __init__(self, fig_path="./output/", perf_data_dir="./perf_log/", metrics_config=None):
        """
        参数:
            fig_path: 图片保存目录 (默认 ./output/)
            perf_data_dir: 存放 *_tester_results_data.npy 的目录 (默认 ./perf_log/)
            metrics_config: 指标配置字典，键为指标名（与 npy 第三维顺序一致），
                            值为包含 'display','unit','ylim','force_natural',
                            'scale','smooth_window','outlier_threshold','outlier_mode' 的字典。
        """
        self.fig_path = fig_path
        self.perf_data_dir = perf_data_dir
        self.metrics_config = metrics_config
        self.data_config = list(metrics_config.keys()) if metrics_config else []
        os.makedirs(fig_path, exist_ok=True)

        # 自动扫描并合并 _para 文件（支持同一算法的多个并行运行结果）
        pattern = os.path.join(perf_data_dir, '*_tester_results_data.npy')
        files = glob.glob(pattern)
        groups = {}
        self.algo_order = {}  # 存储每个算法的排序值

        for fpath in files:
            base = os.path.basename(fpath)
            # 提取原始名字
            if base.endswith('_tester_results_data.npy'):
                raw_name = base[:-len('_tester_results_data.npy')]
            else:
                raw_name = base.split('.')[0]
            # 去除可能的 _para 后缀
            if raw_name.endswith('_para'):
                base_name = raw_name[:-5]
            else:
                base_name = raw_name

            # 提取排序前缀（例如 "0_Proposed" -> 排序0, 算法名 "Proposed"）
            parts = base_name.split('_', 1)
            if len(parts) == 2 and parts[0].isdigit():
                sort_order = int(parts[0])
                algo_name = parts[1]
            else:
                sort_order = None  # 无前缀，排在最后
                algo_name = base_name

            groups.setdefault(algo_name, []).append(fpath)
            if algo_name not in self.algo_order:
                self.algo_order[algo_name] = sort_order
            else:
                if self.algo_order[algo_name] != sort_order:
                    print(f"Warning: Algorithm '{algo_name}' has conflicting sort orders. "
                          f"Keeping first value {self.algo_order[algo_name]}, ignoring {sort_order}.")

        self.data = {}
        self.total_episodes = {}
        for algo_name, file_list in groups.items():
            data_list = [np.load(f) for f in file_list]
            if len(data_list) == 1:
                merged = data_list[0]
            else:
                merged = np.concatenate(data_list, axis=0)
            self.data[algo_name] = merged
            self.total_episodes[algo_name] = merged.shape[0]

        if not self.data:
            print(f"Warning: No files found in {perf_data_dir}")

    # -------------------- 滑动窗口平滑 --------------------
    def _smooth_series(self, data_2d, window):
        if window is None or window <= 1:
            return data_2d.copy()
        smoothed = np.empty_like(data_2d)
        for e in range(data_2d.shape[0]):
            series = data_2d[e, :]
            for i in range(len(series)):
                start = max(0, i - window + 1)
                smoothed[e, i] = np.nanmean(series[start:i+1])
        return smoothed

    # -------------------- 异常值剔除 --------------------
    def _remove_outliers(self, data_2d, threshold, mode='step'):
        if threshold is None:
            return data_2d.copy(), np.sum(~np.isnan(data_2d), axis=0)

        data = data_2d.copy()
        if mode == 'step':
            valid_counts = np.zeros(data.shape[1], dtype=int)
            for s in range(data.shape[1]):
                col = data[:, s]
                valid = ~np.isnan(col)
                if np.sum(valid) == 0:
                    continue
                mean = np.mean(col[valid])
                std = np.std(col[valid])
                if std == 0:
                    valid_counts[s] = np.sum(valid)
                    continue
                z_scores = np.abs((col - mean) / std)
                outlier_mask = z_scores > threshold
                data[outlier_mask, s] = np.nan
                valid_counts[s] = np.sum(~np.isnan(data[:, s]))
            return data, valid_counts

        elif mode == 'global':
            flat_vals = data[~np.isnan(data)]
            if len(flat_vals) == 0:
                return data, np.zeros(data.shape[1], dtype=int)
            mean = np.mean(flat_vals)
            std = np.std(flat_vals)
            if std == 0:
                return data, np.sum(~np.isnan(data), axis=0)
            z_scores = np.abs((data - mean) / std)
            outlier_mask = z_scores > threshold
            data[outlier_mask] = np.nan
            valid_counts = np.sum(~np.isnan(data), axis=0)
            return data, valid_counts
        else:
            raise ValueError(f"Unknown outlier_mode: {mode}")

    # -------------------- 统计计算 --------------------
    def _process_metric_data(self, metric_data, scale, smooth_window, outlier_threshold, outlier_mode):
        data_scaled = metric_data * scale
        data_smoothed = self._smooth_series(data_scaled, smooth_window)
        data_cleaned, valid_counts = self._remove_outliers(data_smoothed, outlier_threshold, outlier_mode)
        return data_cleaned, valid_counts

    def _compute_step_stats(self, metric_data, scale, smooth_window, outlier_threshold, outlier_mode):
        cleaned, valid_counts = self._process_metric_data(
            metric_data, scale, smooth_window, outlier_threshold, outlier_mode
        )
        mean_vals = np.nanmean(cleaned, axis=0)
        std_vals = np.nanstd(cleaned, axis=0)
        return mean_vals, std_vals, valid_counts

    def _compute_global_stats(self, metric_data, scale, smooth_window, outlier_threshold, outlier_mode):
        cleaned, _ = self._process_metric_data(
            metric_data, scale, smooth_window, outlier_threshold, outlier_mode
        )
        valid_vals = cleaned[~np.isnan(cleaned)]
        if len(valid_vals) == 0:
            return np.nan, np.nan
        return np.mean(valid_vals), np.std(valid_vals)

    # -------------------- CSV 导出 --------------------
    def save_statistics_to_csv(self, csv_path='./output/performance_stats.csv'):
        if not self.data:
            return
        headers = ['Algorithm'] + [f'{k}_mean' for k in self.data_config] + [f'{k}_std' for k in self.data_config]
        rows = []
        sorted_algos = sorted(self.data.keys(), key=lambda a: self.algo_order.get(a, float('inf')))
        for algo in sorted_algos:
            data = self.data[algo]
            row = [algo]
            for idx, key in enumerate(self.data_config):
                cfg = self.metrics_config[key]
                scale = cfg.get('scale', 1.0)
                smooth_window = cfg.get('smooth_window', None)
                outlier_threshold = cfg.get('outlier_threshold', cfg.get('zscore_threshold', None))
                outlier_mode = cfg.get('outlier_mode', 'step')
                metric_data = data[:, :, idx]
                mean_val, _ = self._compute_global_stats(
                    metric_data, scale, smooth_window, outlier_threshold, outlier_mode
                )
                row.append(mean_val)
            for idx, key in enumerate(self.data_config):
                cfg = self.metrics_config[key]
                scale = cfg.get('scale', 1.0)
                smooth_window = cfg.get('smooth_window', None)
                outlier_threshold = cfg.get('outlier_threshold', cfg.get('zscore_threshold', None))
                outlier_mode = cfg.get('outlier_mode', 'step')
                metric_data = data[:, :, idx]
                _, std_val = self._compute_global_stats(
                    metric_data, scale, smooth_window, outlier_threshold, outlier_mode
                )
                row.append(std_val)
            rows.append(row)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Statistics saved to {csv_path}")

    # -------------------- 绘图：主指标曲线（不含生还率） --------------------
    def plot_comparison(self):
        if not self.data:
            return

        # 计算每个算法的生还率（用于独立绘图）
        self.survival_ratios = {}
        for algo, data in self.data.items():
            any_valid = np.any(~np.isnan(data), axis=2)
            valid_counts = np.sum(any_valid, axis=0)
            self.survival_ratios[algo] = valid_counts / self.total_episodes[algo]

        # 计算每个指标的处理后统计量
        plot_info = {}
        for algo, data in self.data.items():
            plot_info[algo] = {}
            for idx, key in enumerate(self.data_config):
                cfg = self.metrics_config[key]
                scale = cfg.get('scale', 1.0)
                smooth_window = cfg.get('smooth_window', None)
                outlier_threshold = cfg.get('outlier_threshold', cfg.get('zscore_threshold', None))
                outlier_mode = cfg.get('outlier_mode', 'step')
                metric_data = data[:, :, idx]
                mean_vals, std_vals, _ = self._compute_step_stats(
                    metric_data, scale, smooth_window, outlier_threshold, outlier_mode
                )
                plot_info[algo][key] = (mean_vals, std_vals)

        sorted_algos = sorted(self.data.keys(), key=lambda a: self.algo_order.get(a, float('inf')))

        for idx, key in enumerate(self.data_config):
            cfg = self.metrics_config[key]
            force_natural = cfg.get('force_natural', False)
            ylim = cfg.get('ylim', None)

            fig, ax = plt.subplots(figsize=(10, 6))

            if force_natural:
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])
                ylabel = f"{cfg['display']} ({cfg['unit']})" if cfg['unit'] else cfg['display']
                ax.set_ylabel(ylabel, fontsize=16)

                for i, algo in enumerate(sorted_algos):
                    mean_vals, std_vals = plot_info[algo][key]
                    if np.all(np.isnan(mean_vals)):
                        continue
                    linestyle = LINESTYLES[i % len(LINESTYLES)]
                    line, = ax.plot(mean_vals, label=algo, linewidth=3, linestyle=linestyle)
                    color = line.get_color()
                    ax.fill_between(range(len(mean_vals)),
                                     mean_vals - std_vals,
                                     mean_vals + std_vals,
                                     color=color, alpha=0.1)

            else:
                if ylim is not None:
                    ref_value = ylim[1]
                else:
                    all_vals = []
                    for algo in sorted_algos:
                        mean_vals, std_vals = plot_info[algo][key]
                        all_vals.extend(mean_vals[~np.isnan(mean_vals)])
                        all_vals.extend(std_vals[~np.isnan(std_vals)])
                    ref_value = np.nanmax(all_vals) if all_vals else 1.0

                if ref_value > 0:
                    exponent = int(np.floor(np.log10(ref_value)))
                    if ref_value / (10 ** exponent) < 1:
                        exponent -= 1
                else:
                    exponent = 0

                if exponent == 1:
                    if ylim is not None:
                        ax.set_ylim(ylim[0], ylim[1])
                    ylabel = f"{cfg['display']} ({cfg['unit']})" if cfg['unit'] else cfg['display']
                    ax.set_ylabel(ylabel, fontsize=16)
                    for i, algo in enumerate(sorted_algos):
                        mean_vals, std_vals = plot_info[algo][key]
                        if np.all(np.isnan(mean_vals)):
                            continue
                        linestyle = LINESTYLES[i % len(LINESTYLES)]
                        line, = ax.plot(mean_vals, label=algo, linewidth=3, linestyle=linestyle)
                        color = line.get_color()
                        ax.fill_between(range(len(mean_vals)),
                                         mean_vals - std_vals,
                                         mean_vals + std_vals,
                                         color=color, alpha=0.1)
                else:
                    scale = 10 ** exponent
                    if ylim is not None:
                        ax.set_ylim(ylim[0] / scale, ylim[1] / scale)
                    ylabel = f"{cfg['display']} (×10$^{{{exponent}}}$ {cfg['unit']})" if cfg['unit'] else f"{cfg['display']} (×10$^{{{exponent}}}$)"
                    ax.set_ylabel(ylabel, fontsize=16)
                    for i, algo in enumerate(sorted_algos):
                        mean_vals, std_vals = plot_info[algo][key]
                        if np.all(np.isnan(mean_vals)):
                            continue
                        linestyle = LINESTYLES[i % len(LINESTYLES)]
                        line, = ax.plot(mean_vals / scale, label=algo, linewidth=3, linestyle=linestyle)
                        color = line.get_color()
                        ax.fill_between(range(len(mean_vals)),
                                         (mean_vals - std_vals) / scale,
                                         (mean_vals + std_vals) / scale,
                                         color=color, alpha=0.1)

            ax.set_xlabel('Steps', fontsize=16)
            ax.grid(True)
            ax.legend(fontsize=14, loc='best')

            def sci_formatter(x, pos):
                s = f'{x:.2f}'.rstrip('0').rstrip('.')
                return s if s else '0'
            ax.yaxis.set_major_formatter(FuncFormatter(sci_formatter))
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_path, f"{key}_comparison.png"))
            plt.close()

        print(f"Comparison plots saved to {self.fig_path}")

    # -------------------- 独立绘制生还率曲线 --------------------
    def plot_survival_rates(self):
        """单独绘制所有算法的生还率曲线（随 step 单调递减）"""
        if not hasattr(self, 'survival_ratios') or not self.survival_ratios:
            print("No survival data to plot.")
            return

        sorted_algos = sorted(self.data.keys(), key=lambda a: self.algo_order.get(a, float('inf')))
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, algo in enumerate(sorted_algos):
            ratio = self.survival_ratios[algo]
            if len(ratio) == 0:
                continue
            linestyle = LINESTYLES[i % len(LINESTYLES)]
            ax.plot(range(len(ratio)), ratio, label=algo, linewidth=2, linestyle=linestyle)

        ax.set_xlabel('Steps', fontsize=16)
        ax.set_ylabel('Survival Rate', fontsize=16)
        ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.legend(fontsize=14, loc='best')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, "survival_rate_comparison.png"))
        plt.close()
        print(f"Survival rate plot saved to {self.fig_path}/survival_rate_comparison.png")

    # -------------------- 主流程 --------------------
    def run(self):
        """运行完整流程：导出统计 CSV、绘制各指标对比图、绘制生还率独立图"""
        self.save_statistics_to_csv()
        self.plot_comparison()
        self.plot_survival_rates()


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # ---------- 配置指标（顺序必须与 npy 第三维完全一致） ----------
    metrics_config = {
        'Min Fluent': {
            'display': 'Min Capacity',
            'unit': 'bit/s',
            'ylim': None,
            'force_natural': False,
            'scale': 1.0e6,
            'smooth_window': 10,
            'outlier_threshold': None
        },
        'Sum Fluent': {
            'display': 'Total Capacity',
            'unit': 'bit/s',
            'ylim': None,
            'force_natural': False,
            'scale': 1.0e6,
            'smooth_window': 10,
            'outlier_threshold': None
        },
        'JFI': {
            'display': 'JFI',
            'unit': '',
            'ylim': (0.8, 1.0),
            'force_natural': True,
            'scale': 1.0,
            'smooth_window': 10,
            'outlier_threshold': None
        },
        'Energy': {
            'display': 'Thrust Power',
            'unit': 'W',
            'ylim': None,
            'force_natural': False,
            'scale': 1.0/3,
            'smooth_window': 10,
            'outlier_threshold': None
        },
        'Delay': {
            'display': 'Delay',
            'unit': 's',
            'ylim': (3.3e-3, 5.3e-3),
            'force_natural': False,
            'scale': 1e-6,
            'smooth_window': 10,
            'outlier_threshold': 3.0,
            'outlier_mode': 'global'
        },
        'Min Eff': {
            'display': 'Min Efficiency',
            'unit': 'bit/(s·W)',
            'ylim': None,
            'force_natural': False,
            'scale': 1.0e6,
            'smooth_window': 10,
            'outlier_threshold': None
        },
        'Total Eff': {
            'display': 'Total Efficiency',
            'unit': 'bit/(s·W)',
            'ylim': None,
            'force_natural': False,
            'scale': 1.0e6,
            'smooth_window': 10,
            'outlier_threshold': None
        }
    }

    # ---------- 实例化并运行 ----------
    comparator = PerfComparator(
        fig_path="./output/",
        perf_data_dir="./perf_log/",
        metrics_config=metrics_config
    )
    comparator.run()