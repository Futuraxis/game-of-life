import numpy as np
class GameOfLife:
    def __init__(self, p, rows=50, cols=50, survive_rule=(2,3), birth_rule=(3,), boundary='dead'):
        """
        初始化生命游戏数据结构。
        
        参数:
            p: float, 初始细胞存活概率
            rows, cols: int, 网格行数和列数
            survive_rule: tuple, 存活所需的邻居数（例如 (2,3)）
            birth_rule: tuple, 出生所需的邻居数（例如 (3,)）
            boundary: str, 边界条件，'dead' 表示边界外视为死细胞（也可扩展为 'periodic'）
        """
        self.p = p
        self.rows = rows
        self.cols = cols
        self.survive_rule = survive_rule
        self.birth_rule = birth_rule
        self.boundary = boundary
         # 当前网格状态（0=死，1=活），初始由概率p随机生成
        self.grid = self._initiaize_grid()
         # 当前代数
        self.generation = 0
        # 历史记录：用于保存每代的网格（可选，可根据内存调整）
        self.history = [self.grid.copy()]
        def _initialize_grid(self):
            """根据概率p随机生成初始网格"""
        return (np.random.rand(self.rows, self.cols) < self.p).astype(np.int8)
    def get_current_grid(self):
        return self.grid
    def get_current_generation(self):
        return self.generation
    def get_history(self):
        return self.history
    def get_parameters(self):
        return {
            'p':self.p,
            'rows':self.rows,
            'cols':self.cols,
            'survive_rule':self.survive_rule,
            'birth_rule':self.birth_rule,
            'boundary':self.boundary
        }
    def set_grid(self,new_grid):
        self.grid = new_grid.copy()
        self.generation = 0
        self.history = [self.grid.copy()]
    #作用：允许外部直接指定网格（例如从文件加载），并重置代数和历史记录。这里同样使用 copy() 避免后续修改污染新网格。
    def increment_generation(self):
        self.geneation += 1
    def record(self):
        self.history.append(self.grid.copy())
    def __repr__(self):
        return (f"GameOfLife(p={self.p}, size={self.rows}x{self.cols}, "
            f"gen={self.generation}, rules=({self.survive_rule},{self.birth_rule}))")
        