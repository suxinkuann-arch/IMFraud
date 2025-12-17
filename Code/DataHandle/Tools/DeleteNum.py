import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import os
import datetime


def setup_robust_font():
    """
    更健壮的字体设置，解决数字和符号缺失问题
    """
    try:
        # 清除字体缓存
        fm._rebuild()

        # 设置完整的字体回退链，确保数字和符号能正常显示
        font_families = []

        # 首先尝试添加系统中可能存在的英文字体（确保数字和符号支持）
        english_fonts = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans',
                         'Bitstream Vera Sans', 'Verdana']

        # 然后添加中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
                         'Source Han Sans CN', 'Noto Sans CJK SC']

        # 检查系统中实际可用的字体
        available_fonts = set()
        for font in fm.fontManager.ttflist:
            available_fonts.add(font.name)

        # 构建字体回退链：先英文（保证数字符号），后中文
        for font in english_fonts:
            if font in available_fonts:
                font_families.append(font)
                break
        else:
            # 如果没有找到英文字体，至少添加DejaVu Sans（matplotlib自带）
            font_families.append('DejaVu Sans')

        # 添加可用的中文字体
        chinese_added = False
        for font in chinese_fonts:
            if font in available_fonts:
                font_families.append(font)
                chinese_added = True
                print(f"使用中文字体: {font}")
                break

        if not chinese_added:
            print("警告: 未找到可用的中文字体，将使用英文字体显示")

        plt.rcParams['font.sans-serif'] = font_families
        plt.rcParams['axes.unicode_minus'] = False

        # 测试字体是否支持基本字符
        test_chars = "1234567890+()-"
        print(f"字体设置完成: {font_families}")

    except Exception as e:
        print(f"字体设置过程中出现错误: {e}")
        # 回退到最基本的字体设置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False


# 初始化字体设置
setup_robust_font()


def read_jsonl_files(directory_path):
    """
    读取指定目录下所有jsonl文件，提取input字段长度
    """
    all_lengths = []

    # 匹配目录下所有.jsonl文件
    file_pattern = f"{directory_path}/*.jsonl"
    jsonl_files = glob.glob(file_pattern)

    if not jsonl_files:
        print(f"在目录 {directory_path} 中未找到任何.jsonl文件")
        return all_lengths

    print(f"找到 {len(jsonl_files)} 个JSONL文件")

    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'input' in data and isinstance(data['input'], str):
                            text_length = len(data['input'])
                            all_lengths.append(text_length)
                    except json.JSONDecodeError as e:
                        print(f"文件 {file_path} 第 {line_num} 行JSON解析错误: {e}")
                        continue
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

    return all_lengths


def plot_histogram_with_fit_and_highlight(lengths, save_path=None):
    """
    绘制带拟合曲线和高亮前5%的直方图 - 最终修复版
    """
    if not lengths:
        print("没有有效数据可绘制")
        return

    # 转换为numpy数组便于计算
    lengths_array = np.array(lengths)

    # 计算前5%的阈值
    threshold_5percent = np.percentile(lengths_array, 5)
    print(f"前5%长度阈值: {threshold_5percent:.2f} 字符")

    # 基本统计信息
    print(f"总样本数: {len(lengths_array)}")
    print(f"最小长度: {lengths_array.min()}")
    print(f"最大长度: {lengths_array.max()}")
    print(f"平均长度: {lengths_array.mean():.2f}")
    print(f"中位数: {np.median(lengths_array):.2f}")

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))

    # 定义颜色
    normal_color = 'steelblue'
    highlight_color = 'lightcoral'

    # 绘制直方图
    n, bins, patches = ax.hist(lengths_array, bins=30, density=True, alpha=0.7,
                               color=normal_color, edgecolor='black', linewidth=0.5)

    # 高亮前5%的柱子
    for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
        bin_right = bins[i + 1]
        if bin_right <= threshold_5percent or (bin_left <= threshold_5percent <= bin_right):
            patch.set_facecolor(highlight_color)

    # 拟合正态分布曲线
    mu, std = stats.norm.fit(lengths_array)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label=f'Normal distribution fit (μ={mu:.1f}, σ={std:.1f})')

    # 添加KDE曲线
    sns.kdeplot(lengths_array, color='red', linewidth=2, linestyle='--',
                label='Kernel Density Estimation', ax=ax)

    # 添加前5%阈值线
    ax.axvline(x=threshold_5percent, color='red', linestyle=':',
               linewidth=2, label=f'Top 5% threshold ({threshold_5percent:.1f} chars)')

    # 设置图形属性（使用英文避免字体问题）
    ax.set_xlabel('Input Text Length (characters)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('JSONL File Input Field Length Distribution Analysis\n(Top 5% highlighted in red)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 创建图例说明颜色含义
    legend_elements = [
        Patch(facecolor=normal_color, alpha=0.7, label='Normal range (95%)'),
        Patch(facecolor=highlight_color, alpha=0.7, label='Top 5% (needs attention)'),
        plt.Line2D([0], [0], color='k', linewidth=2, label=f'Normal distribution fit (μ={mu:.1f}, σ={std:.1f})'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Kernel Density Estimation'),
        plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2,
                   label=f'Top 5% threshold ({threshold_5percent:.1f} chars)')
    ]
    ax.legend(handles=legend_elements, loc='best')

    # 添加统计信息文本框（使用英文避免字体问题）
    stats_text = f'''Statistics:
Total samples: {len(lengths_array)}
Min length: {lengths_array.min()}
Max length: {lengths_array.max()}
Mean length: {lengths_array.mean():.2f}
Median: {np.median(lengths_array):.2f}
Top 5% threshold: {threshold_5percent:.2f}'''

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace')

    plt.tight_layout()

    # 保存图形到本地
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"主分析图形已保存到: {save_path}")
    else:
        print("错误: 未指定保存路径")

    # 关闭图形释放内存
    plt.close(fig)

    return threshold_5percent


def detailed_length_analysis(lengths, save_path=None):
    """
    进行详细的长度分布分析 - 修复版
    """
    lengths_array = np.array(lengths)

    # 计算各分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(lengths_array, percentiles)

    print("\n" + "=" * 50)
    print("Detailed Length Distribution Analysis")
    print("=" * 50)

    for p, val in zip(percentiles, percentile_values):
        print(f"{p}% percentile: {val:.1f} characters")

    # 长度区间统计
    bin_ranges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                  1000, np.inf]
    bin_labels = ['0-100', '101-200', '201-300', '301-400', '401-500',
                  '501-600', '601-700', '701-800', '801-900', '901-1000', '1000+']

    bin_counts = []
    for i in range(len(bin_ranges) - 1):
        count = ((lengths_array >= bin_ranges[i]) & (lengths_array < bin_ranges[i + 1])).sum()
        bin_counts.append(count)
        percentage = (count / len(lengths_array)) * 100
        print(f"{bin_labels[i]} chars: {count} items ({percentage:.1f}%)")

    # 绘制分布区间图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(bin_labels, bin_counts, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Length Range (characters)', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Input Field Length Range Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)

    # 在柱子上添加数量标注
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                f'{count}', ha='center', va='bottom', fontsize=9,
                fontfamily='DejaVu Sans')  # 明确指定字体

    plt.tight_layout()

    # 保存详细分析图形
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"详细分析图形已保存到: {save_path}")
    else:
        print("错误: 未指定保存路径")

    # 关闭图形释放内存
    plt.close(fig)


# 主执行函数
def main():
    # 请将这里的路径替换为您的实际目录路径
    directory_path = r"D:\Study\Paper\EI\Data\4Gambling\Anonymous"

    print("开始读取JSONL文件...")
    all_lengths = read_jsonl_files(directory_path)

    if not all_lengths:
        print("没有找到有效数据，程序结束")
        return

    print(f"成功读取 {len(all_lengths)} 个样本的input字段长度")

    # 设置输出目录为输入目录下的子目录
    output_dir = os.path.join(directory_path, "length_analysis_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 生成时间戳用于文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 绘制主要直方图并保存
    main_plot_path = os.path.join(output_dir, f'length_distribution_{timestamp}.png')
    threshold = plot_histogram_with_fit_and_highlight(all_lengths, save_path=main_plot_path)

    # 进行详细分析并保存
    detail_plot_path = os.path.join(output_dir, f'length_distribution_detail_{timestamp}.png')
    detailed_length_analysis(all_lengths, save_path=detail_plot_path)

    # 输出前5%样本的详细信息
    lengths_array = np.array(all_lengths)
    short_samples = lengths_array[lengths_array < threshold]
    print(f"\nTop 5% sample count: {len(short_samples)}")
    print(f"Top 5% sample length range: {short_samples.min()} - {short_samples.max()} characters")

    # 保存统计结果到文件
    report_path = os.path.join(output_dir, f'length_analysis_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("JSONL File Input Field Length Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total samples: {len(all_lengths)}\n")
        f.write(f"Top 5% threshold: {threshold:.2f} characters\n")
        f.write(f"Top 5% sample count: {len(short_samples)}\n")
        f.write(f"Mean length: {np.mean(all_lengths):.2f}\n")
        f.write(f"Median length: {np.median(all_lengths):.2f}\n")
        f.write(f"Main plot saved to: {main_plot_path}\n")
        f.write(f"Detail plot saved to: {detail_plot_path}\n")
        f.write(f"Analysis generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {directory_path}\n")
        f.write(f"Output directory: {output_dir}\n")

    print(f"分析报告已保存到: {report_path}")
    print(f"所有结果文件保存在: {output_dir} 目录中")
    print(f"输入文件目录: {directory_path}")


if __name__ == "__main__":
    main()