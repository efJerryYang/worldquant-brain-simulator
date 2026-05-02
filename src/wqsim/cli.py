from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .alphas import ALPHAS
from .config import load_config
from .simulator import run_simulation

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
    )
    result = run_simulation(cfg, alpha)
    typer.echo(
        f"{result.alpha_name}: {len(result.pnl)} days, final cumulative PnL {result.final_pnl:,.2f}"
    )
    if result.plot_path is not None:
        typer.echo(f"figure: {result.plot_path}")


@app.command()
def alphas() -> None:
    """List available alphas."""
    for name in sorted(ALPHAS):
        typer.echo(name)
