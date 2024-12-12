import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PlotMetrics:
    def __init__(self, data_dir, plots_dir):
        self.data_dir = data_dir
        self.plots_dir = plots_dir
        self.model_name_field = "model_name"
        self.sort_field = "total_model_size_gb"
        self.quant_metrics = [
            "perplexity",
            "response_length",
            "repetition_rate",
            "distinct_2",
            "readability",
            "time_to_first_token",
            "avg_time_per_token",
            "tokens_generated_per_response",
        ]
        self.size_metrics = ["num_model_params", "total_model_size_gb"]
        self.df = self.compile_csv_files_to_df()
        self.df_agg = self.save_df_quant_metrics()
        self.df_size = self.save_df_size_metrics()

        logger = logging.getLogger(__name__)

    def compile_csv_files_to_df(self):
        """Loops through a dir, collects all csv files, and compiles into a single df."""

        # Collect all csv files
        csv_files = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".csv"):
                file_path = os.path.join(self.data_dir, file_name)
                csv_files.append(file_path)

        # No csv files found
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
        df = df.sort_values(by=[self.sort_field])
        df.to_csv(os.path.join(self.plots_dir, "df.csv"), index=False)
        return df

    def save_df_quant_metrics(self):
        """Save an aggregated df with mean/std stats."""
        metrics_dict = {metric: ["mean", "std"] for metric in self.quant_metrics}
        metrics_dict[self.sort_field] = ["mean", "std"]
        df_agg = self.df.groupby(self.model_name_field, as_index=False).agg(metrics_dict)

        # Flatten columns
        df_agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col for col in df_agg.columns
        ]
        df_agg = df_agg.sort_values(by=[self.sort_field + "_mean"])

        # Save df
        df_agg.to_csv(os.path.join(self.plots_dir, "df_agg.csv"), index=False)
        return df_agg

    def save_df_size_metrics(self):
        """Save a df with model size metrics."""
        df_size = self.df.groupby(self.model_name_field, as_index=False)[self.size_metrics].max()
        df_size = df_size.sort_values(by=[self.sort_field])
        df_size.to_csv(os.path.join(self.plots_dir, "df_size.csv"), index=False)
        return df_size

    def plot_violin(self, metric):
        """Plots the distribution of a metric across all models using a violin plot."""
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in the DataFrame.")

        # Plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            data=self.df,
            x=self.model_name_field,
            y=metric,
            palette="muted",
            hue=self.model_name_field,
            legend=False,
        )
        plt.title(metric.replace("_", " ").title(), fontsize=16)
        plt.ylabel("")
        plt.xlabel(self.model_name_field.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.plots_dir, f"{metric}_violin_plot.png")
        plt.savefig(plot_path)

    def plot_size_metric(self, metric):
        """Plots model sizes."""
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in the DataFrame.")

        # Plot
        plt.figure(figsize=(12, 6))

        sns.barplot(
            x=self.model_name_field,
            y=metric,
            hue=self.model_name_field,
            data=self.df_size,
            palette="muted",
            legend=False,
        )

        plt.title(metric.replace("_", " ").title(), fontsize=16)
        plt.ylabel("")
        plt.xlabel(self.model_name_field.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.plots_dir, f"{metric}_size_plot.png")
        plt.savefig(plot_path)

    def plot_all_metrics(self):
        """Plots all metrics and saves output as png."""
        for metric in self.quant_metrics:
            self.plot_violin(metric)

        for metric in self.size_metrics:
            self.plot_size_metric(metric)
