from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .alphas import ALPHAS
from .config import load_config
from .data import load_market_panel
from .simulator import POST_PROCESS_MODES, run_simulation

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    alpha: Annotated[str, typer.Option(help="Alpha name from the registry.")] = "eg_alpha3",
    sample: Annotated[
        str, typer.Option(help="Sample range: test, insample, outsample, latest.")
    ] = "test",
    config: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
    database_path: Annotated[Path | None, typer.Option(help="SQLite market database path.")] = None,
    universe: Annotated[
        str | None, typer.Option(help="Universe selector, for example Top3000.")
    ] = None,
    start_date: Annotated[
        str | None, typer.Option(help="Simulation start date, YYYY-MM-DD.")
    ] = None,
    end_date: Annotated[str | None, typer.Option(help="Data end date, YYYY-MM-DD.")] = None,
    output_dir: Annotated[Path, typer.Option(help="Directory for generated figures.")] = Path(
        "tmp"
    ),
    plot: Annotated[bool, typer.Option(help="Write a cumulative PnL figure.")] = True,
    metrics: Annotated[bool, typer.Option(help="Print deterministic summary metrics.")] = False,
    post_process_mode: Annotated[
        str, typer.Option(help="Portfolio post-processing mode.")
    ] = "legacy",
) -> None:
    """Run one alpha through the batch simulator."""
    cfg = load_config(
        config,
        sample=sample,
        database_path=database_path,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        plot=plot,
        post_process_mode=post_process_mode,
    )
    result = run_simulation(cfg, alpha)
    typer.echo(
        f"{result.alpha_name}: {len(result.pnl)} days, final cumulative PnL {result.final_pnl:,.2f}"
    )
    if result.plot_path is not None:
        typer.echo(f"figure: {result.plot_path}")
    if result.metrics_path is not None:
        typer.echo(f"metrics: {result.metrics_path}")
    if metrics:
        for key in (
            "simulation_start_date",
            "simulation_end_date",
            "traded_days",
            "panel_rows",
            "panel_symbols",
            "post_process_mode",
            "max_drawdown",
            "mean_abs_net_exposure",
            "alpha_nan_count",
            "final_pnl",
            "mean_daily_pnl",
            "std_daily_pnl",
            "mean_turnover",
        ):
            typer.echo(f"{key}: {result.metrics[key]}")


@app.command()
def compare_postprocess(
    alpha: Annotated[str, typer.Option(help="Alpha name from the registry.")] = "eg_alpha3",
    sample: Annotated[
        str, typer.Option(help="Sample range: test, insample, outsample, latest.")
    ] = "insample",
    config: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
    database_path: Annotated[Path | None, typer.Option(help="SQLite market database path.")] = None,
    universe: Annotated[
        str | None, typer.Option(help="Universe selector, for example Top3000.")
    ] = None,
    start_date: Annotated[
        str | None, typer.Option(help="Simulation start date, YYYY-MM-DD.")
    ] = None,
    end_date: Annotated[str | None, typer.Option(help="Data end date, YYYY-MM-DD.")] = None,
) -> None:
    """Compare portfolio post-processing modes on one loaded panel."""
    base_config = load_config(
        config,
        sample=sample,
        database_path=database_path,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        plot=False,
    )
    panel = load_market_panel(base_config)
    typer.echo("mode,final_pnl,max_drawdown,mean_abs_net_exposure,mean_turnover,min_daily_pnl")
    for mode in POST_PROCESS_MODES:
        cfg = load_config(
            config,
            sample=sample,
            database_path=database_path,
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            plot=False,
            post_process_mode=mode,
        )
        result = run_simulation(cfg, alpha, panel)
        typer.echo(
            f"{mode},"
            f"{result.metrics['final_pnl']},"
            f"{result.metrics['max_drawdown']},"
            f"{result.metrics['mean_abs_net_exposure']},"
            f"{result.metrics['mean_turnover']},"
            f"{result.metrics['min_daily_pnl']}"
        )


@app.command()
def alphas() -> None:
    """List available alphas."""
    for name in sorted(ALPHAS):
        typer.echo(name)
