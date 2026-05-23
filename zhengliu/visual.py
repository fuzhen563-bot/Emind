"""
蒸馏可视化仪表盘 v3.0 — 高级终端面板
  pip install rich → 完整体验
  无 Rich        → 简洁文本进度
"""
import time
import threading

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.progress import (
        Progress, BarColumn, TextColumn, SpinnerColumn,
        TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn,
    )
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.style import Style
    from rich.spinner import Spinner
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class DistillDashboard:
    STYLE_GREEN = "bold green"
    STYLE_RED = "bold red"
    STYLE_YELLOW = "yellow"
    STYLE_CYAN = "cyan"
    STYLE_MAGENTA = "magenta"

    def __init__(self, total: int, type_counts: dict):
        self.total = total
        self.type_counts = type_counts
        self.completed = 0
        self.success = 0
        self.fail = 0
        self.quality_fail = 0
        self.by_type = {}        # {type: success_count}
        self.quality_scores = []  # 最近 100 个质量分
        self.recent_samples = []  # 最近 5 条生成样本预览
        self.recent_events = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._live = None
        self._progress = None
        self._task_id = None
        self._type_tasks = {}     # per-type progress tasks
        self.console = Console() if _HAS_RICH else None
        self._paused = False
        self._mode = "SFT"
        self._model = ""

    def set_mode(self, mode: str):
        self._mode = mode

    def set_model(self, model: str):
        self._model = model

    # ======================== Start / Finish ========================

    def start(self):
        if _HAS_RICH and self.console:
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=30, style="green", complete_style="green", finished_style="bright_green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._task_id = self._progress.add_task("[cyan]总进度", total=self.total)
            for t, c in self.type_counts.items():
                tid = self._progress.add_task(f"[dim]{t}", total=c, visible=(c > 0))
                self._type_tasks[t] = tid
            self._live = Live(self._layout(), console=self.console, refresh_per_second=6, transient=False)
            self._live.__enter__()
        else:
            self._simple_header()

    def finish(self):
        if _HAS_RICH and self._live:
            self._live.__exit__(None, None, None)
            self.console.print()
            self._summary_table()

    # ======================== Layout ========================

    def _layout(self):
        l = Layout()
        l.split(
            Layout(self._header(), size=3),
            Layout(name="main"),
            Layout(self._footer(), size=3),
        )
        l["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="center", ratio=3),
            Layout(name="right", ratio=2),
        )
        l["left"].split(
            Layout(Panel(self._progress, title="📊 进度", border_style="green"), size=2 + len(self.type_counts) * 2),
            Layout(Panel(self._type_table(), title="📋 类型", border_style="blue")),
        )
        l["center"].split(
            Layout(Panel(self._sample_preview(), title="📝 最近生成", border_style="cyan"), size=6),
            Layout(Panel(self._events_panel(), title="📜 事件", border_style="bright_black")),
        )
        l["right"].split(
            Layout(Panel(self._stats_table(), title="⚡ 统计", border_style="yellow")),
            Layout(Panel(self._quality_panel(), title="🎯 质量分 (最近)", border_style="magenta")),
        )
        return l

    def _header(self):
        mode_str = f"[bold cyan]zhengliu[/]  [dim]v3.0[/]  │  [yellow]{self._mode}[/]  │  [dim]{self._model or 'API'}[/]"
        if self._paused:
            mode_str += "  [bold red]⏸ 已暂停[/]"
        return Panel(Text(mode_str, justify="center"), box=box.HEAVY, border_style="cyan")

    def _footer(self):
        return Panel(
            Text("Ctrl+C 停止  |  P 暂停/恢复  |  S 导出当前数据", style="dim", justify="center"),
            box=box.MINIMAL,
        )

    # ======================== Panels ========================

    def _type_table(self):
        t = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        t.add_column("类型", style="cyan", width=12)
        t.add_column("完成", justify="right", width=8, style="green")
        t.add_column("失败", justify="right", width=6, style="red")
        for dtype, total in sorted(self.type_counts.items()):
            ok = self.by_type.get(dtype, 0)
            total_done = self.completed
            type_frac = total / max(1, self.total)
            type_done = min(total, int(total_done * type_frac + ok * 0.2))
            fail_est = min(total - ok, int((total_done - ok) * 0.3))
            t.add_row(dtype, f"{ok}/{total}", str(max(0, fail_est)))
        return t

    def _stats_table(self):
        t = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        t.add_column("k", style="dim", width=8)
        t.add_column("v", style="white")
        e = time.time() - self._start_time
        rate = self.completed / max(1, e) * 60
        sr = self.success / max(1, self.completed) * 100
        remaining = (self.total - self.completed) / max(1, rate) * 60 if rate > 0 else 0
        t.add_row("进度", f"{self.completed}/{self.total}")
        t.add_row("成功", f"[green]{self.success}[/]  |  [red]{self.fail}[/] 失败")
        t.add_row("成功率", f"{sr:.0f}%")
        t.add_row("速率", f"{rate:.1f} 条/分")
        t.add_row("已用时", f"{e:.0f}s")
        if remaining > 0:
            t.add_row("剩余", f"{remaining:.0f}s")
        return t

    def _quality_panel(self):
        scores = self.quality_scores[-50:]
        if not scores:
            return Text("等待数据...", style="dim")
        avg = sum(scores) / len(scores)
        mn, mx = min(scores), max(scores)
        bars_per_level = 10
        hist = [0] * bars_per_level
        for s in scores:
            idx = min(bars_per_level - 1, int(s * bars_per_level))
            hist[idx] += 1
        max_h = max(hist) if hist else 1
        lines = [f"均值: [bold]{avg:.2f}[/]  范围: {mn:.1f}~{mx:.1f}  N={len(scores)}"]
        for i in range(bars_per_level - 1, -1, -1):
            lo, hi = i / bars_per_level, (i + 1) / bars_per_level
            bar_w = int(hist[i] / max_h * 16) if max_h else 0
            color = "red" if hi < 0.4 else ("yellow" if hi < 0.7 else "green")
            bar = "█" * bar_w + " " * (16 - bar_w)
            lines.append(f"[{color}]{bar}[/] [{dim]}{lo:.1f}-{hi:.1f}")
        return Text("\n".join(lines))

    def _sample_preview(self):
        samples = self.recent_samples[-5:]
        if not samples:
            return Text("等待生成...", style="dim")
        texts = []
        for i, s in enumerate(reversed(samples)):
            dtype = s.get("type", "?")
            content = s.get("text", "")[:70].replace("\n", " ")
            qs = s.get("quality", None)
            qs_str = f" [{'green' if qs and qs > 0.6 else 'yellow' if qs and qs > 0.3 else 'red'}]{qs:.2f}[/]" if qs is not None else ""
            texts.append(f"[dim]{dtype}[/]{qs_str}  {content}")
        return Text("\n".join(texts))

    def _events_panel(self):
        events = self.recent_events[-10:]
        if not events:
            return Text("等待中...", style="dim")
        colored = []
        for e in events:
            if "✗" in e or "失败" in e:
                colored.append(f"[red]{e}[/]")
            elif "✓" in e or "成功" in e:
                colored.append(f"[green]{e}[/]")
            elif "⚠" in e:
                colored.append(f"[yellow]{e}[/]")
            else:
                colored.append(f"[dim]{e}[/]")
        return Text("\n".join(colored))

    def _summary_table(self):
        e = time.time() - self._start_time
        sr = self.success / max(1, self.total) * 100
        t = Table(title="蒸馏完成", box=box.ROUNDED, title_style="bold green")
        t.add_column("指标", style="cyan")
        t.add_column("值", style="white")
        t.add_row("模式", self._mode)
        t.add_row("模型", self._model or "API")
        t.add_row("总处理", str(self.total))
        t.add_row("成功", f"[green]{self.success}[/]")
        t.add_row("失败", f"[red]{self.fail}[/]")
        t.add_row("质量过滤", str(self.quality_fail))
        t.add_row("成功率", f"{sr:.1f}%")
        t.add_row("用时", f"{e:.1f}s")
        if self.quality_scores:
            t.add_row("平均质量分", f"{sum(self.quality_scores)/len(self.quality_scores):.2f}")
        self.console.print(t)

    # ======================== Fallback ========================

    def _simple_header(self):
        print(f"\n{'='*50}")
        print(f"  zhengliu v3.0  │  {self._mode}  │  {self._model or 'API'}")
        print(f"  {self.total} 条 → Teacher API")
        print(f"{'='*50}")
        for t, c in self.type_counts.items():
            print(f"    {t}: {c}")
        print()

    # ======================== Update ========================

    def update(self, completed=None, success=None, fail=None, quality_fail=None,
               by_type=None, event=None, sample=None, quality_score=None):
        with self._lock:
            if completed is not None:
                self.completed = completed
            if success is not None:
                self.success = success
            if fail is not None:
                self.fail = fail
            if quality_fail is not None:
                self.quality_fail = quality_fail
            if by_type:
                for k, v in by_type.items():
                    self.by_type[k] = self.by_type.get(k, 0) + v
            if event:
                ts = time.strftime("%H:%M:%S")
                self.recent_events.append(f"[{ts}] {event}")
                if len(self.recent_events) > 200:
                    self.recent_events = self.recent_events[-200:]
            if sample:
                self.recent_samples.append(sample)
                if len(self.recent_samples) > 50:
                    self.recent_samples = self.recent_samples[-50:]
            if quality_score is not None:
                self.quality_scores.append(quality_score)
                if len(self.quality_scores) > 200:
                    self.quality_scores = self.quality_scores[-200:]

        if _HAS_RICH and self._live and self._progress:
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=self.completed)
            for t, c in self.type_counts.items():
                if t in self._type_tasks:
                    ok = self.by_type.get(t, 0)
                    self._progress.update(self._type_tasks[t], completed=ok)
            self._live.update(self._layout())
        else:
            if self.completed % 5 == 0 or self.completed == self.total:
                bar_w = 24
                f = int(bar_w * self.completed / max(1, self.total))
                bar = "█" * f + "░" * (bar_w - f)
                rate = self.completed / max(1, time.time() - self._start_time) * 60
                print(f"  [{bar}] {self.completed}/{self.total}  ✓{self.success} ✗{self.fail}  {rate:.1f}条/分")


__all__ = ["DistillDashboard"]
