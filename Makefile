.PHONY: figures

# Regenerate figures from results/metrics_per_model.csv
figures:
	@echo "Generating figures into ./figures/ (using scripts/plot_metrics.py)"
	python3 scripts/plot_metrics.py
