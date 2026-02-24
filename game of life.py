import numpy as np
import os
import json
import time
from copy import deepcopy
from itertools import product
import scipy.signal
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

# 生命游戏 Moore 卷积核
K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])

class GameOfLife:    
    def default_evolve(grid):
        A = grid
        U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
        A = (A & (U==2)) | (U==3)
        return A
    
    def __init__(self, p, rows=50, cols=50, evolve=default_evolve, boundary='dead'):
        self.p = p # 生成概率 p
        self.rows = rows # 网格行数
        self.cols = cols # 网格列数
        self.evolve = evolve
        self.boundary = boundary # 边界条件
        
        self.grid = self._initialize_grid() # 当前网格
        self.generation = 0 # 当前代数
        self.history = [self.grid.copy()]  # 记录初始代 
    
    # 初始化网格
    def _initialize_grid(self):
        return (np.random.rand(self.rows, self.cols) < self.p).astype(np.int8)
    
    # 获得当前网格
    def get_current_grid(self):
        return self.grid
    
    # 获得当前代数
    def get_current_generation(self):
        return self.generation
    
    # 获得历史
    def get_history(self):
        return self.history
    
    # 获得各参数
    def get_parameters(self):
        return {
            'p': self.p,
            'rows': self.rows,
            'cols': self.cols,
            'survive_rule': self.survive_rule,
            'birth_rule': self.birth_rule,
            'boundary': self.boundary
        }
    
    # 设置当前网格
    def set_grid(self, new_grid):
        self.grid = new_grid.copy()
        self.generation = 0
        self.history = [self.grid.copy()]
    
    # 增加代数
    def increment_generation(self):
        # 演化
        self.grid = self.evolve(self.grid)
        self.history.append(self.grid)

        self.generation += 1
    
    # 记录历史
    def record(self):
        """记录当前网格到历史"""
        self.history.append(self.grid.copy())
    
    # 字符串表示
    def __repr__(self):
        return (f"GameOfLife(p={self.p}, size={self.rows}x{self.cols}, "
                f"gen={self.generation}, rules=({self.survive_rule},{self.birth_rule}))")

class Experiment:
    @staticmethod
    def default_evolve_func(game, max_steps=100, single_step=None,
                            interval=1, computation_info=None, use_history=True, seed = 42): #不知道为啥需要加 seed
        """
        通用演化函数：对传入的 `game` 执行最多 `max_steps` 次演化。
        single_step: 单次演化函数，接受并返回 numpy ndarray（2D）。
                     若为 None，则使用 GameOfLife.default_evolve。
        computation_info: dict, e.g. {'device':'CPU'|'GPU', 'parallelism':'single-thread'|'multi-thread', 'workers':int}
        返回: (cell_counts, final_grid)
        """

        # 单步函数不存在则设置为默认值
        if single_step is None:
            single_step = GameOfLife.default_evolve

        # 获取参数，默认值为：CPU 单线程
        comp = computation_info or {'device': 'CPU', 'parallelism': 'single-thread'}
        device = comp.get('device', 'CPU').upper()
        parallel = comp.get('parallelism', 'single-thread')

        # 获取行数和列数
        rows, cols = game.rows, game.cols

        # 计算存活细胞数
        cell_counts = [int(game.grid.sum())]

        # CPU 多线程按行分块计算下一代（带环绕边界）
        def cpu_next(grid):
            if parallel != 'multi-thread':
                return single_step(grid)

            workers = int(comp.get('workers', os.cpu_count() or 1))
            n_workers = min(max(1, workers), rows)
            if n_workers <= 1:
                return single_step(grid)

            # 构建分块区间 [s, e)（行索引）
            chunk_sizes = [(rows // n_workers) + (1 if i < (rows % n_workers) else 0) for i in range(n_workers)]
            starts = []
            cur = 0
            for sz in chunk_sizes:
                starts.append((cur, cur + sz))
                cur += sz

            def compute_chunk(se):
                s, e = se
                # 提取包含上下各一行的子网格（环绕取模）
                idx = [((r) % rows) for r in range(s - 1, e + 1)]
                sub = grid[idx, :]
                U = scipy.signal.convolve2d(sub, K, mode='same', boundary='wrap')
                # 中心区域对应原始 s..e-1 行
                mid_start = 1
                mid_len = e - s
                U_center = U[mid_start: mid_start + mid_len, :]
                sub_center = sub[mid_start: mid_start + mid_len, :]
                next_chunk = ((sub_center & (U_center == 2)) | (U_center == 3)).astype(np.int8)
                return s, next_chunk

            next_grid = np.zeros_like(grid)
            with ThreadPoolExecutor(max_workers=n_workers) as exe:
                futures = [exe.submit(compute_chunk, se) for se in starts if se[0] < se[1]]
                for fut in futures:
                    s, chunk = fut.result()
                    next_grid[s: s + chunk.shape[0], :] = chunk
            return next_grid

        # GPU 版本（使用 torch）
        def gpu_next(grid):
            if not torch.cuda.is_available():
                # Fallback to CPU single-thread
                return single_step(grid)

            t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
            # kernel
            k = torch.tensor(K.astype(np.float32)).unsqueeze(0).unsqueeze(0).to('cuda')
            # 使用 circular pad 再 conv2d
            padded = F.pad(t, (1, 1, 1, 1), mode='circular')
            U = F.conv2d(padded, k)
            A = (t > 0.5)
            U = U.squeeze(0).squeeze(0)
            A = A.squeeze(0).squeeze(0)
            nextA = ((A & (U == 2)) | (U == 3)).to(torch.uint8)
            return nextA.cpu().numpy().astype(np.int8)

        # 主循环
        for step in range(int(max_steps)):
            if device == 'GPU':
                next_grid = gpu_next(game.grid)
            else:
                next_grid = cpu_next(game.grid)

            game.grid = next_grid.copy()
            game.generation += 1
            if use_history and interval and (game.generation % interval == 0):
                game.history.append(game.grid.copy())
            cell_counts.append(int(game.grid.sum()))

        return cell_counts, game.grid

    """
    批量实验管理器。可对多组参数自动运行生命游戏，收集每代细胞数、最终网格等数据。
    新增功能：记录每次实验的计算方式（CPU/GPU、单线程/多线程）。
    """
    def __init__(self, configs, evolve_func=default_evolve_func, save_dir='experiment_data', computation_info=None):
        """
        参数:
            configs : list of dict
                每个字典包含一次实验的所有参数，至少包括 'p'。可含：
                - GameOfLife 参数：'rows', 'cols', 'survive_rule', 'birth_rule', 'boundary'
                - 演化参数：'max_steps', 'seed', 'computation_info' 等
            evolve_func : callable
                演化函数，接受 (game, **kwargs) 并返回 (cell_counts, final_grid)。
            save_dir : str
                保存实验结果的目录。
            computation_info : dict, optional
                默认的计算方式信息，包含键 'device' 和 'parallelism'。
                例如 {'device': 'CPU', 'parallelism': 'single-thread'}。
                如果为 None，则使用默认值 {'device': 'CPU', 'parallelism': 'single-thread'}。
                如果某个配置中指定了 'computation_info'，则优先使用配置中的值。
        """
        self.configs = configs
        self.evolve_func = evolve_func
        self.save_dir = save_dir
        self.results = []          # 存储每次实验的结果字典
        os.makedirs(save_dir, exist_ok=True)

        # 设置默认计算方式信息
        if computation_info is None:
            self.computation_info = {'device': 'CPU', 'parallelism': 'single-thread'}
        else:
            self.computation_info = computation_info

    def run_all(self, **common_kwargs):
        """
        运行所有配置的实验。common_kwargs 是所有实验共享的演化参数（如 max_steps）。
        返回 results 列表。
        """
        self.results = []
        for i, cfg in enumerate(self.configs):
            print(f"实验 {i+1}/{len(self.configs)}: p={cfg.get('p')}, "
                  f"size={cfg.get('rows',50)}x{cfg.get('cols',50)}")

            # 1. 提取 GameOfLife 参数，补全默认值
            # 根据计算方式选取evolve函数

            game_params = {k: cfg.get(k, default) 
                           for k, default in [('p', None), ('rows',50), ('cols',50),
                                              ('evolve',GameOfLife.default_evolve),
                                              ('boundary','dead')] if k in cfg or k=='p'}
            if 'p' not in game_params:
                raise ValueError("每个配置必须包含 'p'")

            # 2. 设置随机种子（如果提供）
            if 'seed' in cfg:
                np.random.seed(cfg['seed'])

            # 3. 创建 GameOfLife 实例
            game = GameOfLife(**game_params)

            # 4. 准备演化参数（合并公共参数和当前配置独有参数）
            evolve_params = common_kwargs.copy()
            evolve_params.update({k: cfg[k] for k in ['max_steps', 'seed', 'interval'] 
                                  if k in cfg})

            # 5. 确定本次实验的计算方式信息（优先使用配置中的，否则使用默认）
            comp_info = cfg.get('computation_info', self.computation_info).copy()

            # 6. 运行演化
            start = time.time()
            # 将本次实验的计算信息传入演化函数
            cell_counts, final_grid = self.evolve_func(game, computation_info=comp_info, **evolve_params)
            elapsed = time.time() - start

            # 7. 记录结果（增加 computation_info 字段）
            result = {
                'config': deepcopy(cfg),                # 原始配置
                'game_params': game_params,              # 实际使用的 GameOfLife 参数
                'computation_info': comp_info,            # 新增：计算方式信息
                'initial_count': cell_counts[0] if cell_counts else 0,
                'final_count': cell_counts[-1] if cell_counts else 0,
                'cell_counts': cell_counts,              # 每代细胞数列表
                'final_grid': final_grid.copy(),          # 最终网格
                'history': game.get_history(),            # 所有代的网格（若内存敏感可移除）
                'generations': game.get_current_generation(),
                'runtime': elapsed
            }
            self.results.append(result)

        print(f"所有 {len(self.configs)} 个实验完成。")
        return self.results

    def save_results(self, exp_name=None, save_history=False):
        """
        保存实验结果：元数据（JSON） + 最终网格（NPZ）。
        元数据中现在包含 computation_info 字段。
        """
        if exp_name is None:
            exp_name = time.strftime("%Y%m%d_%H%M%S")

        # 元数据（不含网格）
        meta = []
        for res in self.results:
            meta.append({
                'config': res['config'],
                'game_params': res['game_params'],
                'computation_info': res['computation_info'],   # 新增
                'initial_count': res['initial_count'],
                'final_count': res['final_count'],
                'cell_counts': res['cell_counts'],
                'generations': res['generations'],
                'runtime': res['runtime']
            })
        meta_path = os.path.join(self.save_dir, f"{exp_name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"元数据已保存至 {meta_path}")

        # 最终网格
        grids = np.array([res['final_grid'] for res in self.results])
        grids_path = os.path.join(self.save_dir, f"{exp_name}_final_grids.npz")
        np.savez_compressed(grids_path, grids=grids)
        print(f"最终网格已保存至 {grids_path}")

        # 可选：保存历史网格（每个实验单独文件）
        if save_history:
            hist_dir = os.path.join(self.save_dir, f"{exp_name}_history")
            os.makedirs(hist_dir, exist_ok=True)
            for i, res in enumerate(self.results):
                hist_array = np.array(res['history'])  # shape: (n_gen, rows, cols)
                np.save(os.path.join(hist_dir, f"exp{i}_history.npy"), hist_array)
            print(f"历史网格已保存至 {hist_dir}")

    @staticmethod
    def generate_configs(param_grid, fixed_params=None):
        """
        根据参数网格生成所有配置组合。
        param_grid : dict {参数名: 取值列表}
        fixed_params : dict 所有配置共用的固定参数
        返回 configs 列表。
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        configs = []
        for combo in product(*values):
            cfg = dict(zip(keys, combo))
            if fixed_params:
                cfg.update(fixed_params)
            configs.append(cfg)
        return configs

    def to_dataframe(self):
        """
        将结果转换为 pandas DataFrame（需安装 pandas）。
        现在也会包含 computation_info 的列。
        """
        try:
            import pandas as pd
        except ImportError:
            print("请安装 pandas: pip install pandas")
            return None
        rows = []
        for res in self.results:
            row = {}
            # 展开配置
            for k, v in res['config'].items():
                if isinstance(v, tuple):
                    v = str(v)
                row[f'cfg_{k}'] = v
            # 添加计算方式信息
            comp = res.get('computation_info', {})
            row['device'] = comp.get('device', 'unknown')
            row['parallelism'] = comp.get('parallelism', 'unknown')
            # 添加统计
            row['initial'] = res['initial_count']
            row['final'] = res['final_count']
            row['generations'] = res['generations']
            row['runtime'] = res['runtime']
            rows.append(row)
        return pd.DataFrame(rows)

# ================= 可视化功能 =================
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML  # 用于在 Jupyter Notebook 中显示动画
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 未安装，可视化功能将不可用。请运行 'pip install matplotlib' 安装。")

class Visualizer:
    """生命游戏可视化工具类"""

    @staticmethod
    def plot_grid(grid, title=None, ax=None, cmap='gray', show=True):
        """
        绘制单个网格（二维数组）
        :param grid: 二维 numpy 数组 (0/1)
        :param title: 图像标题
        :param ax: matplotlib 轴对象，若为 None 则新建
        :param cmap: 颜色映射，默认 'gray' 显示黑白
        :param show: 是否立即显示图像
        :return: 绘制的轴对象
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib 未安装，无法绘图。")
            return None
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return ax

    @staticmethod
    def plot_cell_counts(counts, title="Cell Count Over Generations", ax=None, show=True):
        """
        绘制细胞数量随时间的变化曲线
        :param counts: 每代细胞数量列表
        :param title: 图表标题
        :param ax: 轴对象
        :param show: 是否立即显示
        :return: 轴对象
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib 未安装，无法绘图。")
            return None
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(counts, linewidth=1.5)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Number of live cells")
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)
        if show:
            plt.show()
        return ax

    @staticmethod
    def animate_history(history, interval=200, repeat=True, figsize=(6,6), cmap='gray'):
        """
        生成历史网格的动画
        :param history: 网格历史列表，每个元素为二维 numpy 数组 (代, rows, cols)
        :param interval: 帧间隔（毫秒）
        :param repeat: 是否循环播放
        :param figsize: 图像尺寸
        :param cmap: 颜色映射
        :return: matplotlib.animation.FuncAnimation 对象
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib 未安装，无法生成动画。")
            return None
        if len(history) == 0:
            print("历史记录为空，无法生成动画。")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xticks([])
        ax.set_yticks([])
        img = ax.imshow(history[0], cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        title = ax.set_title(f"Generation 0")

        def update(frame):
            img.set_array(history[frame])
            title.set_text(f"Generation {frame}")
            return [img, title]

        ani = animation.FuncAnimation(fig, update, frames=len(history),
                                      interval=interval, repeat=repeat, blit=True)
        plt.close(fig)  # 防止额外显示静态图像
        return ani

    @staticmethod
    def plot_final_grids(results, max_cols=4, figsize=(12,12), cmap='gray'):
        """
        批量显示多个实验的最终网格（子图排列）
        :param results: Experiment.run_all() 返回的结果列表
        :param max_cols: 每行最多显示多少个子图
        :param figsize: 整图尺寸
        :param cmap: 颜色映射
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib 未安装，无法绘图。")
            return
        n = len(results)
        if n == 0:
            print("无实验结果可显示。")
            return
        cols = min(max_cols, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, res in enumerate(results):
            ax = axes[i]
            grid = res['final_grid']
            ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            p = res['config'].get('p', '?')
            ax.set_title(f"p={p}\ngen={res['generations']}\ncount={res['final_count']}", fontsize=10)
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_animation_in_notebook(ani):
        """
        在 Jupyter Notebook 中显示动画（需要 IPython.display）
        :param ani: matplotlib.animation.FuncAnimation 对象
        :return: HTML 对象
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib 未安装。")
            return None
        try:
            from IPython.display import HTML
            return HTML(ani.to_jshtml())
        except ImportError:
            print("IPython 未安装，无法在 Notebook 中显示动画。")
            return ani

# ================= 使用示例 =================
if __name__ == '__main__':
    # 1. 手动定义几个配置，可以在配置中指定 computation_info
    configs = [
        {'p': 0.1, 'rows': 300, 'cols': 300, 'max_steps': 50, 'seed': 42,
         'computation_info': {'device': 'CPU', 'parallelism': 'single-thread'}},
        {'p': 0.3, 'rows': 300, 'cols': 300, 'max_steps': 150, 'seed': 42,
         'computation_info': {'device': 'CPU', 'parallelism': 'multi-thread'}},
        {'p': 0.5, 'rows': 300, 'cols': 300, 'max_steps': 450, 'seed': 42},
        # 第三个未指定，将使用 Experiment 默认的计算方式
    ]

    print("加载参数...")
    # 2. 创建实验对象，传入默认计算方式
    exp = Experiment(configs, save_dir='batch_results',
                     computation_info={'device': 'GPU', 'parallelism': 'single-thread'})

    print("开始进行实验")
    # 3. 运行所有实验
    results = exp.run_all(max_steps=100)

    # 4. 查看结果中的 computation_info
    for r in results:
        print(f"p={r['config']['p']}, 计算方式: {r['computation_info']}")

    # 5. 保存结果
    exp.save_results('test_002')

    # 6. 转换为 DataFrame 查看
    df = exp.to_dataframe()
    if df is not None:
        print(df[['cfg_p', 'device', 'parallelism', 'initial', 'final']])
    
        # ========== 可视化示例 ==========
    if MATPLOTLIB_AVAILABLE:
        # 1. 绘制单个实验的最终网格
        if results:
            print("\n显示第一个实验的最终网格：")
            Visualizer.plot_grid(results[0]['final_grid'], title=f"Final Grid (p={results[0]['config']['p']})")

        # 2. 绘制细胞数量曲线
        if results and 'cell_counts' in results[0]:
            print("\n显示第一个实验的细胞数量变化：")
            Visualizer.plot_cell_counts(results[0]['cell_counts'], title="Cell Counts Over Time")

        # 3. 批量显示最终网格
        print("\n显示所有实验的最终网格对比：")
        Visualizer.plot_final_grids(results, max_cols=3)

        # 4. 动画示例（使用第一个实验的历史，若 history 不为空）
        if results and len(results[0]['history']) > 1:
            print("\n生成第一个实验的历史动画（将保存为 GIF 或显示在 Notebook 中）...")
            ani = Visualizer.animate_history(results[0]['history'], interval=100)
            # 若要保存为 GIF，可取消下一行注释
            # ani.save('game_of_life.gif', writer='pillow', fps=10)
            plt.show()  # 在某些环境中需要调用 plt.show() 才能显示动画
    else:
        print("请安装 matplotlib 以使用可视化功能。")
