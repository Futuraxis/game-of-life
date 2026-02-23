import numpy as np
import os
import json
import time
from copy import deepcopy
from itertools import product

class GameOfLife:
    def __init__(self, p, rows=50, cols=50, survive_rule=(2,3), birth_rule=(3,), boundary='dead'):
        self.p = p
        self.rows = rows
        self.cols = cols
        self.survive_rule = survive_rule
        self.birth_rule = birth_rule
        self.boundary = boundary
        
        self.grid = self._initialize_grid()
        self.generation = 0
        self.history = [self.grid.copy()]  # 记录初始代
    
    def _initialize_grid(self):
        return (np.random.rand(self.rows, self.cols) < self.p).astype(np.int8)
    
    def get_current_grid(self):
        return self.grid
    
    def get_current_generation(self):
        return self.generation
    
    def get_history(self):
        return self.history
    
    def get_parameters(self):
        return {
            'p': self.p,
            'rows': self.rows,
            'cols': self.cols,
            'survive_rule': self.survive_rule,
            'birth_rule': self.birth_rule,
            'boundary': self.boundary
        }
    
    def set_grid(self, new_grid):
        self.grid = new_grid.copy()
        self.generation = 0
        self.history = [self.grid.copy()]
    
    def increment_generation(self):
        self.generation += 1
    
    def record(self):
        """记录当前网格到历史"""
        self.history.append(self.grid.copy())
    
    def __repr__(self):
        return (f"GameOfLife(p={self.p}, size={self.rows}x{self.cols}, "
                f"gen={self.generation}, rules=({self.survive_rule},{self.birth_rule}))")

class Experiment:
    """
    批量实验管理器。可对多组参数自动运行生命游戏，收集每代细胞数、最终网格等数据。
    新增功能：记录每次实验的计算方式（CPU/GPU、单线程/多线程）。
    """
    def __init__(self, configs, evolve_func, save_dir='experiment_data', computation_info=None):
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
            game_params = {k: cfg.get(k, default) 
                           for k, default in [('p', None), ('rows',50), ('cols',50),
                                              ('survive_rule',(2,3)), ('birth_rule',(3,)),
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
            cell_counts, final_grid = self.evolve_func(game, **evolve_params)
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


# ================= 使用示例 =================
if __name__ == '__main__':
    # 你需要先实现一个真正的演化函数
    def my_evolve(game, max_steps=100, **kwargs):
        """
        示例演化函数（必须替换为真实规则）。
        这里只演示接口，实际应包含邻居计数和更新逻辑。
        """
        counts = [game.grid.sum()]
        for step in range(max_steps):
            # 请在此处添加你的 update 代码
            # new_grid = update(game.grid)  # 需要实现 update
            # game.grid = new_grid
            # game.increment_generation()
            # game.record()
            # counts.append(game.grid.sum())
            # 若稳定可提前 break
            pass
        return counts, game.grid

    # 1. 手动定义几个配置，可以在配置中指定 computation_info
    configs = [
        {'p': 0.1, 'rows': 30, 'cols': 30, 'max_steps': 50, 'seed': 42,
         'computation_info': {'device': 'CPU', 'parallelism': 'single-thread'}},
        {'p': 0.3, 'rows': 30, 'cols': 30, 'max_steps': 50, 'seed': 42,
         'computation_info': {'device': 'CPU', 'parallelism': 'multi-thread'}},
        {'p': 0.5, 'rows': 30, 'cols': 30, 'max_steps': 50, 'seed': 42},
        # 第三个未指定，将使用 Experiment 默认的计算方式
    ]

    # 2. 创建实验对象，传入默认计算方式
    exp = Experiment(configs, evolve_func=my_evolve, save_dir='batch_results',
                     computation_info={'device': 'GPU', 'parallelism': 'single-thread'})

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
