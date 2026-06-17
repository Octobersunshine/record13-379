import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ChartConfig:
    chart_type: str
    title: str
    x_data: np.ndarray
    y_data: np.ndarray
    x_label: str = ""
    y_label: str = ""
    color: str = "#4C72B0"
    grid: bool = True
    marker: str = "o"
    bar_width: float = 0.6
    alpha: float = 0.8
    size: int = 50
    label: str = ""
    secondary_y_data: Optional[np.ndarray] = None
    secondary_color: str = "#DD8452"
    secondary_label: str = ""


class MultiChartService:
    def __init__(self, figsize: Optional[Tuple[int, int]] = None, dpi: int = 150,
                 suptitle: str = "", style: str = "seaborn-v0_8-whitegrid",
                 min_subplot_width: float = 4.5, min_subplot_height: float = 3.2,
                 max_cols: int = 5, max_rows: int = 8,
                 target_aspect: float = 1.4, auto_font_scale: bool = True):
        self._figsize = figsize
        self.dpi = dpi
        self.suptitle = suptitle
        self.style = style
        self.min_subplot_width = min_subplot_width
        self.min_subplot_height = min_subplot_height
        self.max_cols = max_cols
        self.max_rows = max_rows
        self.target_aspect = target_aspect
        self.auto_font_scale = auto_font_scale
        self._configs: List[ChartConfig] = []
        self._grid_shape: Optional[Tuple[int, int]] = None

    @property
    def figsize(self) -> Tuple[int, int]:
        if self._figsize is not None:
            return self._figsize
        rows, cols = self._infer_grid()
        n = len(self._configs)
        if n <= 6:
            w_pad, h_pad = 2.5, 2.0
        elif n <= 12:
            w_pad, h_pad = 2.2, 1.8
        elif n <= 20:
            w_pad, h_pad = 2.0, 1.6
        else:
            w_pad, h_pad = 1.8, 1.4
        cell_w = self.min_subplot_width
        cell_h = self.min_subplot_height
        if n > 12:
            shrink = 1.0 - min(0.25, (n - 12) * 0.01)
            cell_w *= shrink
            cell_h *= shrink
        width = cols * cell_w + w_pad
        height = rows * cell_h + h_pad
        if self.suptitle:
            height += 0.6
        return (max(6, int(np.ceil(width))), max(4, int(np.ceil(height))))

    @figsize.setter
    def figsize(self, value: Tuple[int, int]):
        self._figsize = value

    def set_grid(self, rows: int, cols: int) -> "MultiChartService":
        self._grid_shape = (rows, cols)
        return self

    def add_chart(self, config: ChartConfig) -> "MultiChartService":
        self._configs.append(config)
        return self

    def _calc_font_scale(self) -> float:
        if not self.auto_font_scale:
            return 1.0
        n = len(self._configs)
        if n <= 4:
            return 1.0
        if n <= 6:
            return 0.95
        if n <= 9:
            return 0.88
        if n <= 12:
            return 0.82
        if n <= 16:
            return 0.76
        if n <= 20:
            return 0.70
        if n <= 25:
            return 0.64
        if n <= 30:
            return 0.58
        return max(0.45, 0.58 - (n - 30) * 0.01)

    def _infer_grid(self) -> Tuple[int, int]:
        if self._grid_shape:
            return self._grid_shape
        n = len(self._configs)
        if n <= 0:
            return (1, 1)

        preset = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (2, 4),
            8: (2, 4),
            9: (3, 3),
            10: (2, 5),
            11: (3, 4),
            12: (3, 4),
            13: (3, 5),
            14: (3, 5),
            15: (3, 5),
            16: (4, 4),
            17: (4, 5),
            18: (3, 6),
            19: (4, 5),
            20: (4, 5),
            21: (3, 7),
            22: (4, 6),
            23: (4, 6),
            24: (4, 6),
            25: (5, 5),
            26: (4, 7),
            27: (5, 6),
            28: (4, 7),
            29: (5, 6),
            30: (5, 6),
        }
        if n in preset:
            rows, cols = preset[n]
            rows = min(rows, self.max_rows)
            cols = min(cols, self.max_cols)
            if rows * cols < n:
                pass
            else:
                return (rows, cols)

        best_ratio = float("inf")
        best_grid = (self.max_rows, self.max_cols)
        for cols in range(1, self.max_cols + 1):
            rows = int(np.ceil(n / cols))
            if rows > self.max_rows:
                continue
            actual_ratio = (cols * self.min_subplot_width) / (rows * self.min_subplot_height)
            ratio_dev = abs(actual_ratio - self.target_aspect)
            waste = rows * cols - n
            landscape_bonus = 0.12 if cols >= rows else 0.0
            portrait_penalty = 0.25 if rows > cols * 1.2 else 0.0
            score = ratio_dev + 0.05 * waste - landscape_bonus + portrait_penalty
            if score < best_ratio:
                best_ratio = score
                best_grid = (rows, cols)

        return best_grid

    def _calc_gridspec_kw(self, rows: int, cols: int) -> dict:
        n = len(self._configs)
        if n <= 4:
            hspace, wspace = 0.38, 0.32
            left, right, top, bottom = 0.06, 0.96, 0.93, 0.07
        elif n <= 8:
            hspace, wspace = 0.42, 0.34
            left, right, top, bottom = 0.055, 0.965, 0.93, 0.065
        elif n <= 12:
            hspace, wspace = 0.48, 0.36
            left, right, top, bottom = 0.05, 0.97, 0.94, 0.06
        elif n <= 20:
            hspace, wspace = 0.55, 0.40
            left, right, top, bottom = 0.045, 0.975, 0.945, 0.055
        else:
            hspace, wspace = 0.62, 0.45
            left, right, top, bottom = 0.04, 0.98, 0.95, 0.05
        return {"hspace": hspace, "wspace": wspace,
                "left": left, "right": right,
                "top": top, "bottom": bottom}

    def _render_chart(self, ax: plt.Axes, cfg: ChartConfig, font_scale: float = 1.0):
        title_fs = int(12 * font_scale)
        label_fs = int(10 * font_scale)
        tick_fs = int(9 * font_scale)
        legend_fs = int(9 * font_scale)
        marker_size = max(3, int(5 * font_scale))
        line_width = max(1.2, 2 * font_scale)

        if cfg.chart_type == "line":
            ax.plot(cfg.x_data, cfg.y_data, color=cfg.color, marker=cfg.marker,
                    linewidth=line_width, markersize=marker_size,
                    label=cfg.label or cfg.title)
            if cfg.secondary_y_data is not None:
                ax.plot(cfg.x_data, cfg.secondary_y_data, color=cfg.secondary_color,
                        marker="s", linewidth=line_width, markersize=marker_size,
                        label=cfg.secondary_label)
        elif cfg.chart_type == "bar":
            ax.bar(cfg.x_data, cfg.y_data, width=cfg.bar_width,
                   color=cfg.color, alpha=cfg.alpha, label=cfg.label or cfg.title,
                   edgecolor="white", linewidth=0.5)
            if cfg.secondary_y_data is not None:
                ax.bar(cfg.x_data, cfg.secondary_y_data, width=cfg.bar_width * 0.4,
                       color=cfg.secondary_color, alpha=cfg.alpha,
                       label=cfg.secondary_label, edgecolor="white", linewidth=0.5)
        elif cfg.chart_type == "scatter":
            scatter_size = max(20, int(cfg.size * font_scale * font_scale))
            ax.scatter(cfg.x_data, cfg.y_data, c=cfg.color, s=scatter_size,
                       alpha=cfg.alpha, edgecolors="white", linewidth=0.5,
                       label=cfg.label or cfg.title)
        else:
            raise ValueError(f"Unsupported chart_type: {cfg.chart_type}")

        title_pad = max(6, int(10 * font_scale))
        ax.set_title(cfg.title, fontsize=title_fs, fontweight="bold", pad=title_pad)
        if cfg.x_label:
            ax.set_xlabel(cfg.x_label, fontsize=label_fs)
        if cfg.y_label:
            ax.set_ylabel(cfg.y_label, fontsize=label_fs)
        ax.grid(cfg.grid, linestyle="--", alpha=0.4)
        if cfg.label or cfg.secondary_label:
            ax.legend(fontsize=legend_fs, loc="best", framealpha=0.9)
        ax.tick_params(labelsize=tick_fs)

    def render(self, output_path: str = "multi_chart.png") -> str:
        if not self._configs:
            raise ValueError("No charts added. Use add_chart() first.")

        try:
            plt.style.use(self.style)
        except OSError:
            pass

        rows, cols = self._infer_grid()
        total_cells = rows * cols
        if total_cells < len(self._configs):
            import warnings
            warnings.warn(
                f"Grid {rows}x{cols}={total_cells} cells is insufficient for "
                f"{len(self._configs)} charts. Extra charts will be dropped. "
                f"Consider increasing max_cols or max_rows."
            )
        n_display = min(total_cells, len(self._configs))

        figsize = self.figsize
        font_scale = self._calc_font_scale()
        gridspec_kw = self._calc_gridspec_kw(rows, cols)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=self.dpi,
                                 gridspec_kw=gridspec_kw)
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i in range(n_display):
            self._render_chart(axes_flat[i], self._configs[i], font_scale=font_scale)

        for j in range(n_display, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if self.suptitle:
            suptitle_fs = max(8, int(16 * font_scale))
            fig.suptitle(self.suptitle, fontsize=suptitle_fs,
                         fontweight="bold", y=0.985)

        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                fig.tight_layout(rect=[0, 0, 1, 0.97] if self.suptitle else None)
            except Exception:
                pass

        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path


def build_demo_charts() -> List[ChartConfig]:
    np.random.seed(42)
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    revenue = np.array([120, 135, 148, 162, 155, 170, 185, 198, 210, 225, 240, 260])
    cost = np.array([80, 90, 95, 105, 100, 110, 120, 130, 135, 145, 155, 165])

    categories = np.arange(5)
    cat_labels = ["A", "B", "C", "D", "E"]
    sales = np.array([340, 280, 420, 190, 350])
    returns = np.array([30, 25, 45, 15, 35])

    x_scatter = np.random.uniform(10, 100, 80)
    y_scatter = 2.5 * x_scatter + np.random.normal(0, 30, 80)

    x_scatter2 = np.random.uniform(20, 90, 60)
    y_scatter2 = -1.2 * x_scatter2 + np.random.normal(0, 25, 60) + 150

    profit = revenue - cost
    growth_rate = np.diff(revenue) / revenue[:-1] * 100
    growth_months = months[1:]

    configs = [
        ChartConfig(
            chart_type="line", title="Monthly Revenue vs Cost",
            x_data=months, y_data=revenue, x_label="Month", y_label="Amount (K$)",
            color="#4C72B0", label="Revenue", secondary_y_data=cost,
            secondary_color="#DD8452", secondary_label="Cost"
        ),
        ChartConfig(
            chart_type="bar", title="Sales & Returns by Category",
            x_data=categories, y_data=sales, x_label="Category", y_label="Units",
            color="#55A868", label="Sales", secondary_y_data=returns,
            secondary_color="#C44E52", secondary_label="Returns"
        ),
        ChartConfig(
            chart_type="scatter", title="Feature Correlation Analysis",
            x_data=x_scatter, y_data=y_scatter, x_label="Feature X", y_label="Feature Y",
            color="#8172B3", size=60
        ),
        ChartConfig(
            chart_type="line", title="Monthly Profit Trend",
            x_data=months, y_data=profit, x_label="Month", y_label="Profit (K$)",
            color="#CCB974", marker="D"
        ),
        ChartConfig(
            chart_type="bar", title="Month-over-Month Growth Rate (%)",
            x_data=growth_months, y_data=growth_rate, x_label="Month", y_label="Growth (%)",
            color="#64B5CD"
        ),
        ChartConfig(
            chart_type="scatter", title="Anomaly Detection",
            x_data=x_scatter2, y_data=y_scatter2, x_label="Metric A", y_label="Metric B",
            color="#C44E52", size=45
        ),
    ]
    return configs


def generate_random_charts(count: int) -> List[ChartConfig]:
    np.random.seed(123)
    chart_types = ["line", "bar", "scatter"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#CCB974", "#64B5CD", "#E45756"]
    configs = []
    for i in range(count):
        ct = chart_types[i % 3]
        x = np.arange(1, 11) if ct != "scatter" else np.random.uniform(0, 100, 50)
        y = np.random.randint(20, 100, size=len(x)) if ct != "scatter" \
            else 1.5 * x + np.random.normal(0, 20, 50)
        has_secondary = np.random.random() > 0.6
        secondary = (np.random.randint(10, 60, size=len(x))
                     if has_secondary and ct != "scatter" else None)
        cfg = ChartConfig(
            chart_type=ct,
            title=f"Chart {i+1}: {ct.capitalize()} Plot",
            x_data=x,
            y_data=y,
            x_label="X Axis",
            y_label="Y Axis",
            color=colors[i % len(colors)],
            secondary_y_data=secondary,
            secondary_color=colors[(i + 2) % len(colors)],
            label="Series A" if has_secondary else "",
            secondary_label="Series B" if has_secondary else "",
            marker=["o", "s", "D", "^"][i % 4]
        )
        configs.append(cfg)
    return configs


def test_layout_scaling():
    test_cases = [3, 5, 6, 7, 9, 10, 12, 15, 18, 20, 25, 30]
    print("n   grid    figsize     subplot_size (w×h)")
    print("-" * 55)
    for n in test_cases:
        service = MultiChartService(
            dpi=120,
            suptitle=f"Auto-Layout Test: {n} Charts"
        )
        for cfg in generate_random_charts(n):
            service.add_chart(cfg)
        rows, cols = service._infer_grid()
        fs = service.figsize
        cell_w = round(fs[0] / cols, 1)
        cell_h = round(fs[1] / rows, 1)
        output = service.render(f"multi_chart_{n:02d}.png")
        print(f"{n:2d}  {rows}×{cols:<4}  {fs[0]:3d}×{fs[1]:<3d}   "
              f"{cell_w}×{cell_h}")
    print(f"\nAll {len(test_cases)} layout tests rendered successfully.")


if __name__ == "__main__":
    service = MultiChartService(
        dpi=150,
        suptitle="Business Analytics Dashboard"
    )

    for cfg in build_demo_charts():
        service.add_chart(cfg)

    output = service.render("multi_chart.png")
    print(f"Default 6-chart saved to: {output}")

    print("\n--- Testing layout scaling with various chart counts ---")
    test_layout_scaling()
