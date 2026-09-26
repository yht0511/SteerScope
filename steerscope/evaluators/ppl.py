from .alpaca import AlpacaEvaluator


class PerplexityEvaluator(AlpacaEvaluator):
    compute_perplexity = True

    def render_report(self, result, output_dir=None):
        return self._render_curve_report(
            result,
            {"perplexity": "Perplexity"},
            output_dir,
            columns=1,
            y_scale="log",
            y_axis_label="Perplexity (log scale)",
        )
    
    def __str__(self):
        return 'PerplexityEvaluator'
    
    def compute_metrics(self, data, write_to_dir=None):
        data = data.copy()
        metrics = {
            "perplexity": [],
            "strength": [],
            "factor": []
        }
        
        # group by factor only and compute means
        grouped = data.groupby("factor")
        for factor, group in grouped:
            column = f"{self.model_name}_perplexity"
            if column not in group:
                raise KeyError(
                    f"Perplexity inference for '{self.model_name}' did not produce '{column}'."
                )
            perplexity = group[column].mean()
            metrics["perplexity"].append(perplexity)
            metrics["factor"].append(factor)
            if f"{self.model_name}_strength" in group.columns:
                strength = group[f"{self.model_name}_strength"].mean()
                metrics["strength"].append(strength)
        return metrics
