"""CLI entry point — four subcommands for the full pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from senticrank.config import load_config
from senticrank.logging_setup import setup_logging

app = typer.Typer(name="senticrank", add_completion=False)
logger = logging.getLogger(__name__)


def _get_config(config_path: str = "configs/default.yaml"):
    setup_logging()
    return load_config(Path(config_path))


@app.command()
def split_data(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Stratified train/val/test split → data/interim/."""
    from senticrank.data.loader import load_master_dataset
    from senticrank.data.splitter import split_dataset

    cfg = _get_config(config_path)
    df = load_master_dataset(cfg.data.raw_path)
    split_dataset(
        df,
        interim_dir=cfg.data.interim_dir,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        stratify_columns=cfg.split.stratify_by,
        seed=cfg.seed,
    )
    typer.echo("Split complete. Files in data/interim/")


@app.command()
def train_predictor(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Train all models from config on train.csv, evaluate on val.csv, save best."""
    from senticrank.star_predictor import Trainer

    cfg = _get_config(config_path)
    trainer = Trainer(cfg)
    results = trainer.train_all()

    header = f"{'Model':<20} {'accuracy':>8} {'bal_acc':>8} {'f1_macro':>8} {'mae':>6} {'qwk':>6}"
    typer.echo("\n=== TRAINING RESULTS (val set) ===")
    typer.echo(header)
    for name, m in results.items():
        typer.echo(
            f"{name:<20} {m['accuracy']:>8.3f} {m['balanced_accuracy']:>8.3f}"
            f" {m['f1_macro']:>8.3f} {m['mae']:>6.3f} {m['qwk']:>6.3f}"
        )

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    typer.echo(f"\nBest model: {best_name} (f1_macro={results[best_name]['f1_macro']:.3f} on val)")


@app.command()
def evaluate_predictor(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Evaluate all saved models on test.csv, save metrics.json."""
    from senticrank.star_predictor import Evaluator

    cfg = _get_config(config_path)
    evaluator = Evaluator(cfg)
    df = evaluator.evaluate_on_test()

    typer.echo("\n=== TEST RESULTS (held-out test set) ===")
    header = f"{'Model':<20} {'accuracy':>8} {'bal_acc':>8} {'f1_macro':>8} {'mae':>6} {'qwk':>6}"
    typer.echo(header)
    for _, row in df.iterrows():
        typer.echo(
            f"{row['model']:<20} {row['accuracy']:>8.3f} {row['balanced_accuracy']:>8.3f}"
            f" {row['f1_macro']:>8.3f} {row['mae']:>6.3f} {row['qwk']:>6.3f}"
        )

    best_name = (cfg.star_predictor.output_model_dir and
                 (Path(cfg.star_predictor.output_model_dir) / "best_model.txt"))
    if isinstance(best_name, Path) and best_name.exists():
        bname = best_name.read_text().strip()
        best_row = df[df["model"] == bname]
        if not best_row.empty:
            m = best_row.iloc[0]
            typer.echo(f"\n=== CONFUSION MATRIX ({bname} on test) ===")
            import json
            cm_file = Path(cfg.star_predictor.output_model_dir) / f"{bname}_confusion_matrix.json"
            if cm_file.exists():
                cm = json.loads(cm_file.read_text())
                classes = sorted(cm.keys(), key=int)
                header_cm = "          " + "".join(f"  pred_{c}" for c in classes)
                typer.echo(header_cm)
                for true_c in classes:
                    row_vals = "".join(f"  {cm[true_c].get(c, 0):>6}" for c in classes)
                    typer.echo(f"true_{true_c}   {row_vals}")


@app.command()
def predict_stars(
    text: str = typer.Argument(..., help="Review text to predict star rating for"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Quick smoke test: predict star rating for a single text input."""
    import joblib

    cfg = _get_config(config_path)
    model_path = Path(cfg.star_predictor.output_model_dir) / "best_model.joblib"
    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}. Run train-predictor first.")
        raise typer.Exit(1)
    predictor = joblib.load(model_path)
    pred = predictor.predict([text])[0]
    proba = predictor.predict_proba([text])[0]
    typer.echo(f"Predicted: {pred}⭐")
    typer.echo("Probabilities: " + "  ".join(f"{i+1}⭐={p:.3f}" for i, p in enumerate(proba)))


@app.command()
def detect_fakes(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Score full dataset for fake reviews; writes dataset_with_fake_scores.csv."""
    import json
    from senticrank.data.loader import load_master_dataset
    from senticrank.fake_detector import FakeReviewDetector

    cfg = _get_config(config_path)
    model_path = Path(cfg.star_predictor.output_model_dir) / "best_model.joblib"
    if not model_path.exists():
        typer.echo(f"Model not found at {model_path}. Run train-predictor first.")
        raise typer.Exit(1)

    df = load_master_dataset(cfg.data.raw_path)
    detector = FakeReviewDetector(model_path, cfg.fake_detector)
    scored = detector.score_dataset(df, text_col=cfg.data.text_column)

    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_with_fake_scores.csv"
    scored.to_csv(out_path, index=False)

    counts = scored["fake_category"].value_counts()
    total = len(scored)
    typer.echo(f"\nTotal reviews: {total:,}")
    for cat in ("clean", "suspicious", "very_suspicious", "uncertain"):
        n = counts.get(cat, 0)
        typer.echo(f"  {cat}: {n:,} ({n/total*100:.1f}%)")
    typer.echo(f"\nSaved → {out_path}")


@app.command()
def analyze_fakes(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Print fake-review statistics and save fake_analysis.json + top_fake_examples.csv."""
    import json
    from senticrank.fake_detector import summarize_fakes, find_patterns, top_fake_examples

    cfg = _get_config(config_path)
    scored_path = Path(cfg.data.processed_dir) / "dataset_with_fake_scores.csv"
    if not scored_path.exists():
        typer.echo("Scored dataset not found. Run detect-fakes first.")
        raise typer.Exit(1)

    import pandas as pd
    df = pd.read_csv(scored_path, low_memory=False)

    summary = summarize_fakes(df)
    patterns = find_patterns(df, text_col=cfg.data.text_column)
    examples = top_fake_examples(df, n=20)

    total = summary["total_reviews"]
    typer.echo(f"\nTotal reviews: {total:,}")
    for cat, n in summary["category_counts"].items():
        typer.echo(f"  {cat}: {n:,} ({summary['category_pct'].get(cat, 0):.1f}%)")

    typer.echo("\nSuspicious by product category (top 10):")
    for cat, n in list(summary["suspicious_by_product_category"].items())[:10]:
        typer.echo(f"  {cat}: {n}")

    typer.echo("\nSuspicious by star rating:")
    for star, n in summary["suspicious_by_star_rating"].items():
        typer.echo(f"  {star}⭐: {n}")

    avg = patterns["avg_text_length"]
    typer.echo(f"\nAvg text length — clean: {avg['clean']} chars, fake: {avg['fake']} chars")
    if patterns["helpful_votes"]:
        hv = patterns["helpful_votes"]
        typer.echo(f"Helpful votes — clean mean: {hv['clean_mean']}, fake mean: {hv['fake_mean']}")
        typer.echo(f"Fake reviews with 0 helpful votes: {hv['fake_zero_pct']}%")

    multi = patterns.get("authors_with_3plus_fakes", {})
    if multi:
        typer.echo(f"\nAuthors with ≥3 fake reviews: {len(multi)}")

    out_dir = Path(cfg.fake_detector.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = out_dir / "fake_analysis.json"
    analysis_path.write_text(json.dumps({**summary, **patterns}, indent=2, ensure_ascii=False))

    examples_path = out_dir / "top_fake_examples.csv"
    examples.to_csv(examples_path, index=False)

    typer.echo(f"\nSaved → {analysis_path}")
    typer.echo(f"Saved → {examples_path}")


@app.command()
def filter_dataset_cmd(
    mode: str = typer.Option("conservative", help="'conservative' or 'aggressive'"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Filter scored dataset to remove suspicious reviews."""
    import pandas as pd
    from senticrank.fake_detector import filter_dataset

    cfg = _get_config(config_path)
    scored_path = Path(cfg.data.processed_dir) / "dataset_with_fake_scores.csv"
    if not scored_path.exists():
        typer.echo("Scored dataset not found. Run detect-fakes first.")
        raise typer.Exit(1)

    df = pd.read_csv(scored_path, low_memory=False)
    result = filter_dataset(df, processed_dir=cfg.data.processed_dir, mode=mode)

    clean = result["clean"]
    fakes = result["fakes"]
    typer.echo(f"\nMode: {mode}")
    typer.echo(f"  Clean: {len(clean):,} rows → data/processed/dataset_clean.csv")
    typer.echo(f"  Removed: {len(fakes):,} rows → data/processed/dataset_fakes.csv")


@app.command()
def rank(
    include_fakes: bool = typer.Option(False, "--include-fakes", help="Rank on unfiltered data for comparison"),
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Run SenticRank Fuzzy+TOPSIS ranking engine on clean dataset."""
    import json
    import pandas as pd
    from senticrank.ranking import SenticRankSystem

    cfg = _get_config(config_path)
    scored_path = Path(cfg.data.processed_dir) / "dataset_with_fake_scores.csv"
    if not scored_path.exists():
        typer.echo("Scored dataset not found. Run detect-fakes first.")
        raise typer.Exit(1)

    df = pd.read_csv(scored_path, low_memory=False)
    system = SenticRankSystem(cfg.ranking)

    if include_fakes:
        typer.echo("\n[COMPARISON MODE] Running on unfiltered data (fakes included)…")
        df_all = df.copy()
        df_all["fake_category"] = "clean"
        products_with = system.run(df_all)
        out_path = system.save(products_with, Path(cfg.ranking.output_dir), suffix="_with_fakes")

        typer.echo("\n[MAIN] Running on filtered data (fakes removed)…")
        products_clean = system.run(df)
        system.save(products_clean, Path(cfg.ranking.output_dir))

        typer.echo("\n=== Top-10 comparison: with fakes vs without fakes ===")
        top_clean = products_clean.nsmallest(10, "sentic_rank_overall")[["product_name", "category", "sentic_rank_overall"]].reset_index(drop=True)
        top_with = products_with.nsmallest(10, "sentic_rank_overall")[["product_name", "category", "sentic_rank_overall"]].reset_index(drop=True)
        typer.echo(f"\n{'Rank':<5} {'Without fakes':<40} {'With fakes':<40}")
        for i in range(10):
            c = f"{top_clean.iloc[i]['product_name'][:38]}" if i < len(top_clean) else ""
            w = f"{top_with.iloc[i]['product_name'][:38]}" if i < len(top_with) else ""
            typer.echo(f"{i+1:<5} {c:<40} {w:<40}")
        products = products_clean
    else:
        products = system.run(df)
        system.save(products, Path(cfg.ranking.output_dir))

    fakes_removed = products["num_fakes_removed"].sum()
    rho = products.attrs.get("spearman_rho", float("nan"))
    cr = products.attrs.get("ahp_cr", float("nan"))
    ahp_w = products.attrs.get("ahp_weights", [])

    typer.echo(f"\n{'='*55}")
    typer.echo(f"Products ranked:    {len(products):,}")
    typer.echo(f"Fakes removed:      {fakes_removed:,}")
    typer.echo(f"AHP CR:             {cr:.4f}  {'✓ OK' if cr < 0.10 else '⚠ INCONSISTENT'}")
    typer.echo(f"AHP weights:        {[round(w, 3) for w in ahp_w]}")
    typer.echo(f"Spearman ρ vs stars:{rho:.4f}")
    typer.echo(f"{'='*55}")
    typer.echo(f"\nSaved → {cfg.ranking.output_dir}/senticrank_product_rankings.csv")


@app.command()
def run_all(
    config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
) -> None:
    """Full pipeline: split → train → evaluate → detect-fakes → filter → rank."""
    import json
    from pathlib import Path as P

    typer.echo("\n" + "="*55)
    typer.echo("SenticRank V2 — Full Pipeline")
    typer.echo("="*55)

    ctx = typer.Context(run_all)

    typer.echo("\n[1/6] Splitting data…")
    split_data(config_path=config_path)

    typer.echo("\n[2/6] Training star predictor…")
    train_predictor(config_path=config_path)

    typer.echo("\n[3/6] Evaluating on test set…")
    evaluate_predictor(config_path=config_path)

    typer.echo("\n[4/6] Detecting fake reviews…")
    detect_fakes(config_path=config_path)

    typer.echo("\n[5/6] Filtering dataset…")
    filter_dataset_cmd(mode="conservative", config_path=config_path)

    typer.echo("\n[6/6] Running SenticRank engine…")
    rank(include_fakes=False, config_path=config_path)

    cfg = _get_config(config_path)
    rankings_path = P(cfg.ranking.output_dir) / "senticrank_product_rankings.csv"
    import pandas as pd
    if rankings_path.exists():
        results = pd.read_csv(rankings_path)
        typer.echo("\n" + "="*55)
        typer.echo("FINAL SUMMARY")
        typer.echo("="*55)
        typer.echo(f"Total products ranked: {len(results):,}")
        typer.echo(f"Total fakes removed:   {results['num_fakes_removed'].sum():,}")
        typer.echo(f"\nTop-5 overall:")
        top5 = results.nsmallest(5, "sentic_rank_overall")[
            ["sentic_rank_overall", "product_name", "category", "senticrank_score_100", "avg_star_rating"]
        ]
        for _, row in top5.iterrows():
            typer.echo(
                f"  #{int(row['sentic_rank_overall']):<3} {row['product_name'][:40]:<42}"
                f" [{row['category']}]  score={row['senticrank_score_100']:.1f}  stars={row['avg_star_rating']:.2f}"
            )


if __name__ == "__main__":
    app()
