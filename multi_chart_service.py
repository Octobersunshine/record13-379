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
    def __init__(self, figsize: Tuple[int, int] = (16, 10), dpi: int = 150,
                 suptitle: str = "", style: str = "seaborn-v0_8-whitegrid"):
        self.figsize = figsize
        self.dpi = dpi
        self.suptitle = suptitle
        self.style = style
        self._configs: List[ChartConfig] = []
        self._grid_shape: Optional[Tuple[int, int]] = None

    def set_grid(self, rows: int, cols: int) -> "MultiChartService":
        self._grid_shape = (rows, cols)
        return self

    def add_chart(self, config: ChartConfig) -> "MultiChartService":
        self._configs.append(config)
        return self

    def _infer_grid(self) -> Tuple[int, int]:
        if self._grid_shape:
            return self._grid_shape
        n = len(self._configs)
        if n <= 1:
            return (1, 1)
        if n <= 2:
            return (1, 2)
        if n <= 4:
            return (2, 2)
        if n <= 6:
            return (2, 3)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return (rows, cols)

    def _render_chart(self, ax: plt.Axes, cfg: ChartConfig):
        if cfg.chart_type == "line":
            ax.plot(cfg.x_data, cfg.y_data, color=cfg.color, marker=cfg.marker,
                    linewidth=2, markersize=5, label=cfg.label or cfg.title)
            if cfg.secondary_y_data is not None:
                ax.plot(cfg.x_data, cfg.secondary_y_data, color=cfg.secondary_color,
                        marker="s", linewidth=2, markersize=5,
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
            ax.scatter(cfg.x_data, cfg.y_data, c=cfg.color, s=cfg.size,
                       alpha=cfg.alpha, edgecolors="white", linewidth=0.5,
                       label=cfg.label or cfg.title)
        else:
            raise ValueError(f"Unsupported chart_type: {cfg.chart_type}")

        ax.set_title(cfg.title, fontsize=12, fontweight="bold", pad=10)
        if cfg.x_label:
            ax.set_xlabel(cfg.x_label, fontsize=10)
        if cfg.y_label:
            ax.set_ylabel(cfg.y_label, fontsize=10)
        ax.grid(cfg.grid, linestyle="--", alpha=0.4)
        if cfg.label or cfg.secondary_label:
            ax.legend(fontsize=9, loc="best")
        ax.tick_params(labelsize=9)

    def render(self, output_path: str = "multi_chart.png") -> str:
        if not self._configs:
            raise ValueError("No charts added. Use add_chart() first.")

        try:
            plt.style.use(self.style)
        except OSError:
            pass

        rows, cols = self._infer_grid()
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize, dpi=self.dpi,
                                 constrained_layout=True)
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for i, cfg in enumerate(self._configs):
            self._render_chart(axes_flat[i], cfg)

        for j in range(len(self._configs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        if self.suptitle:
            fig.suptitle(self.suptitle, fontsize=16, fontweight="bold", y=1.02)

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


if __name__ == "__main__":
    service = MultiChartService(
        figsize=(18, 11),
        dpi=150,
        suptitle="Business Analytics Dashboard"
    )
    service.set_grid(2, 3)

    for cfg in build_demo_charts():
        service.add_chart(cfg)

    output = service.render("multi_chart.png")
    print(f"Chart saved to: {output}")
