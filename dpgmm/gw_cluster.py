import h5py
import numpy as np
import glob
import sys
from .dpgmm import DPGMM  # 导入自定义的 DPGMM 模型类
from utils.prog_bar import ProgBar  # 进度条工具
from sklearn.decomposition import PCA  # 用于降维
from sklearn.cluster import KMeans  # K-Means 聚类算法
from sklearn.mixture import GaussianMixture  # 高斯混合模型 (GMM)
from scipy.fft import fft, fftfreq  # FFT 工具
from scipy.signal import welch, hilbert, find_peaks  # 信号分析工具
from sklearn.preprocessing import StandardScaler  # 数据标准化工具

# ------------------------------
# 读取引力波数据
# ------------------------------
def load_strain_data(file_path):
    """
    从 HDF5 文件中加载引力波应变数据。
    """
    with h5py.File(file_path, 'r') as f:
        strain_data = f['strain']['Strain'][()]  # 提取 'Strain' 数据
    return strain_data

# ------------------------------
# 增强的频域和时域特征提取
# ------------------------------
def extract_advanced_features(data, sample_rate=16384):
    """
    从信号数据中提取频域和时域特征。
    """
    N = len(data)  # 获取信号的长度

    # 计算 FFT 幅度谱和频率分量
    yf = np.abs(fft(data)[:N // 2])  # 只取正频率部分
    xf = fftfreq(N, 1 / sample_rate)[:N // 2]  # 对应的频率

    # 1. 基本频域特征
    main_freq = xf[np.argmax(yf)]  # 主频
    max_amplitude = np.max(yf)  # 最大幅度
    mean_amplitude = np.mean(yf)  # 平均幅度
    std_amplitude = np.std(yf)  # 幅度的标准差

    # 2. 前 5 个最高频率幅度值
    top_5_amplitudes = np.sort(yf)[-5:]

    # 3. 频段能量比例：将频谱划分为 3 段，计算每段的平均能量
    band_1_energy = np.mean(yf[:N // 6])  # 频段 1 的平均能量
    band_2_energy = np.mean(yf[N // 6:N // 3])  # 频段 2 的平均能量
    band_3_energy = np.mean(yf[N // 3:N // 2])  # 频段 3 的平均能量

    # 4. 使用 Welch 方法计算功率谱密度 (PSD) 特征
    freqs, psd = welch(data, sample_rate)  # 计算 PSD
    psd_mean = np.mean(psd)  # PSD 的均值
    psd_max = np.max(psd)  # PSD 的最大值

    # 5. 使用 Hilbert 变换计算瞬时频率
    analytic_signal = hilbert(data)  # 希尔伯特变换
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # 解包裹相位
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * sample_rate)  # 计算瞬时频率
    inst_freq_mean = np.mean(instantaneous_frequency)  # 瞬时频率的均值

    # 6. 时域特征：峰值幅度、RMS（均方根值）和零交叉率
    peaks, _ = find_peaks(data)  # 查找峰值
    peak_amplitude = np.max(data[peaks]) if len(peaks) > 0 else 0  # 峰值幅度
    rms_value = np.sqrt(np.mean(data**2))  # 均方根值 (RMS)
    zero_crossings = np.sum(np.diff(np.sign(data)) != 0)  # 零交叉率

    # 返回所有特征作为一个 NumPy 数组
    return np.concatenate((
        [main_freq, max_amplitude, mean_amplitude, std_amplitude, psd_mean, psd_max,
         inst_freq_mean, peak_amplitude, rms_value, zero_crossings],
        top_5_amplitudes,
        [band_1_energy, band_2_energy, band_3_energy]
    ))

# ------------------------------
# 加载数据并提取特征
# ------------------------------
# 自动加载匹配的 HDF5 文件
file_paths = glob.glob('H-H1_GWOSC_16KHZ_*.hdf5')
print(f"HDF5 files found: {file_paths}")

# 从每个文件中提取特征
strain_data_list = [load_strain_data(fp) for fp in file_paths]
features = np.array([extract_advanced_features(data) for data in strain_data_list])
print(f"Extracted feature shape: {features.shape}")

# ------------------------------
# 数据标准化和降维
# ------------------------------
scaler = StandardScaler()  # 初始化标准化工具
features_scaled = scaler.fit_transform(features)  # 标准化特征数据

# 使用 PCA 将数据降维到 9 维
print("Reducing data dimensions with PCA...")
pca = PCA(n_components=9)  # 将数据降到 9 维
reduced_data = pca.fit_transform(features_scaled)
print(f"Reduced data shape: {reduced_data.shape}")

# ------------------------------
# DPGMM 模型训练
# ------------------------------
print('Training DPGMM model...')
model = DPGMM(9)  # 初始化 DPGMM 模型，组件数量为 9

# 将降维后的数据添加到 DPGMM 模型中
for feat in reduced_data:
    model.add(feat)

# 设置模型的先验参数和浓度参数
model.setPrior()
model.setConcGamma(2.0, 0.5)

# 求解模型
p = ProgBar()
iters = model.solveGrow()  # 运行模型迭代
del p  # 删除进度条
print(f'Solved DPGMM model with {iters} iterations')

# 获取分类结果
probs = model.stickProb(reduced_data)  # 获取每个样本的类别概率
dpgmm_labels = probs.argmax(axis=1)  # 找到概率最高的类别
print(f'DPGMM Categorization: {dpgmm_labels}')

# ------------------------------
# K-Means 和 GMM 聚类
# ------------------------------
print('Training K-Means model...')
kmeans = KMeans(n_clusters=3, random_state=42).fit(reduced_data)  # K-Means 聚类
kmeans_labels = kmeans.labels_  # 获取 K-Means 分类标签
print(f'K-Means Categorization: {kmeans_labels}')

print('Training GMM model...')
gmm = GaussianMixture(n_components=3, random_state=42).fit(reduced_data)  # GMM 聚类
gmm_labels = gmm.predict(reduced_data)  # 获取 GMM 分类标签
print(f'GMM Categorization: {gmm_labels}')

# ------------------------------
# 比较分类结果
# ------------------------------
print("\nComparison of clustering results:")
print(f"DPGMM: {dpgmm_labels}")
print(f"K-Means: {kmeans_labels}")
print(f"GMM: {gmm_labels}")
