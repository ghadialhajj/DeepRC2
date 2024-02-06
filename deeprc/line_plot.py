import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import plotly.figure_factory as ff
import dill as pkl

colors = {"LW": "#4daf4a", "LW̅": "#984ea3", "HW̅": "#e41a1c", "HW": "#377eb8"}
color_ids = {"LW": 0, "LW̅": 1, "HW̅": 2, "HW": 3}


class Subplot_from_raw():
    def __init__(self, n_bins=400, factor=1, n_witneses=50):
        self.allowed_models = ['Vanilla', 'FAE', 'TE']
        self.ranks = ['Best', 'Worst']
        self.trace_names = ['HW', 'LW', 'LW̅', 'HW̅']
        self.save_dir = f"/storage/ghadia/DeepRC2/deeprc/results/Attentions/{n_witneses}"
        self.n_bins = n_bins
        self.factor = factor
        self.n_witnesses = n_witneses

    def check_json(self, data: dict):
        assert all([key_model in self.allowed_models for key_model in data.keys()])
        assert all([key_rank in self.ranks for val_model in data.values() for key_rank in val_model.keys()])
        assert all(
            [key_trace in self.trace_names for val_model in data.values() for val_rank in val_model.values()
             for key_trace in val_rank.keys()])
        assert all(
            [type(val_trace) is np.ndarray for val_model in data.values() for val_rank in val_model.values()
             for val_trace in val_rank.values()])

    def preprocess_distplots(self, all_distplots: dict):
        for model_name in self.allowed_models:
            for run_name in self.ranks:
                hists = all_distplots[model_name][run_name]["hists"]
                bins = all_distplots[model_name][run_name]["bins"]
                # normalize hists to sum to 1 using numpy
                hists = {trace: hists[trace] / np.sum(hists[trace]) * 100 for trace in hists}

                bins = {trace: np.array(bins[trace]).reshape((self.n_bins, -1)).mean(axis=1) for trace in bins}
                hists = {trace: hists[trace].reshape((self.n_bins, -1)).sum(axis=1) for trace in hists}

                # if model_name == "Vanilla" and run_name == "Worst":
                #     hists = self.spread_hists2(hists, 1.5)
                hists = {trace: np.log(hists[trace]) for trace in hists}
                all_distplots[model_name][run_name]["hists"] = hists
                all_distplots[model_name][run_name]["bins"] = bins

    def plotting(self, data, type: str = "hist"):
        self.preprocess_distplots(data) if type == "distplot" else data
        x_min, x_max, y_min, y_max = self.get_all_hists_range(data)
        # bins = np.linspace(x_min, x_max, self.n_bins)
        y_min = -10
        y_max += 0.1

        # Create a 3x2 subplot grid
        rows, cols = 2, len(data.keys())
        # Create a 3x3 subplot grid with y-axis title
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=self.allowed_models,
                            horizontal_spacing=0.02, vertical_spacing=0.02,
                            shared_yaxes=True, shared_xaxes=False,
                            y_title="Percentage", x_title="Attention value")  # Set y-axis title directly here
        # Populate the subplot grid with the individual figures
        for j, model_name in enumerate(self.allowed_models):
            for i, rank_name in enumerate(self.ranks):
                index = i * cols + j
                for trace in self.trace_names:
                    fig.add_trace(
                        go.Scatter(x=data[model_name][rank_name]["bins"][trace],
                                   y=data[model_name][rank_name]["hists"][trace],
                                   mode='lines',
                                   name=trace,
                                   line=dict(width=2, color=colors[trace]),
                                   showlegend=index == 2,
                                   legendgroup=trace), row=i + 1, col=j + 1,
                    )

                    # Show x-axis ticks only for the bottom-most row
                    fig.update_xaxes(showticklabels=i == rows - 1, row=i + 1, col=j + 1, range=[x_min, x_max])
                    # Show y-axis ticks only for the leftmost column
                    fig.update_yaxes(showticklabels=j == 0, row=i + 1, col=j + 1, secondary_y=False,
                                     range=[y_min, y_max], title_text=rank_name if j == 0 else None, )
                    # type="log")
            # Update layout and show the combined figure
            fig.update_layout(height=500, width=800, showlegend=True)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',  # Set background color (here, it's set to transparent)
            })
        # Update layout to label figures
        fig.write_image(f"{self.save_dir}/SVG/Attention_{type}_plots_{self.n_witnesses}.svg")
        fig.write_html(f"{self.save_dir}/HTML/Attention_{type}_plots_{self.n_witnesses}.html")
        fig.show()

    def get_range(self, x_data, y_data):
        min_x, max_x = np.min(x_data), np.max(x_data)
        min_y, max_y = np.min(y_data), np.max(y_data)
        return min_x, max_x, min_y, max_y

    def collect_json(self, names_to_keep: dict):

        assert [val_model['Best'] != val_model['Worst'] for val_model in names_to_keep.values()]
        json_dir = f"{self.save_dir}/JSON"
        model_names = [folder for folder in os.listdir(json_dir) if
                       os.path.isdir(f"{json_dir}/{folder}")]

        all_json_data = {model_name: {} for model_name in model_names}

        for model_name in model_names:
            for file in os.listdir(f"{json_dir}/{model_name}"):
                name = next((rank for rank, rank_name in names_to_keep[model_name].items() if file[:-5] == rank_name),
                            None)
                with open(f"{json_dir}/{model_name}/{file}", 'r') as sfh:
                    json_file = json.load(sfh)
                    json_data = json_file["data"]
                    json_data = {d["name"]: np.array(d["x"]) for d in json_data}
                    all_json_data[model_name][name] = json_data
        self.check_json(all_json_data)
        return all_json_data

    def make_histograms(self, all_json_data: dict):
        model_names = all_json_data.keys()

        all_hists = {model_name: {run_name: {} for run_name in all_json_data[model_name]} for model_name in
                     model_names}

        for model_name, json_data in all_json_data.items():
            for run_name, run_data in json_data.items():
                pdf = {}
                bins = {}
                for class_name, class_data in run_data.items():
                    n_bins = self.n_bins * self.factor if (
                            model_name == "Vanilla" and run_name == "Worst") else self.n_bins
                    hist, r_bins = np.histogram(class_data, bins=n_bins, density=True)
                    pdf[class_name] = hist
                    bins[class_name] = r_bins
                pmf = {class_name: pdf[class_name] / np.sum(pdf[class_name]) * 100 for class_name in pdf}
                all_hists[model_name][run_name]["hists"] = pmf
                all_hists[model_name][run_name]["bins"] = bins

        return all_hists

    def make_distplots(self, all_json_data: dict):
        path = f"/storage/ghadia/DeepRC2/deeprc/results/Attentions/{self.n_witnesses}/all_distplots.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pkl.load(f)
        else:
            model_names = all_json_data.keys()

            all_hists = {model_name: {run_name: {} for run_name in all_json_data[model_name]} for model_name in
                         model_names}

            for model_name, json_data in all_json_data.items():
                for run_name, run_data in json_data.items():
                    hist_data = list(run_data.values())
                    group_labels = list(run_data.keys())
                    fig = ff.create_distplot(hist_data, group_labels, histnorm='density', show_hist=False,
                                             show_curve=True, show_rug=False)

                    hists, bins = self.get_dict_from_fig(fig)
                    all_hists[model_name][run_name]["hists"] = hists
                    all_hists[model_name][run_name]["bins"] = bins

            with open(path, 'wb') as f:
                pkl.dump(all_hists, f)

        return all_hists

    def get_dict_from_fig(self, fig):
        hists = {trace.name: trace.y for trace in fig.data}
        bins = {trace.name: trace.x for trace in fig.data}
        return hists, bins

    def spread_hists(self, hists, bins):
        new_bin_values = {k: np.repeat(bins[k], 2) for k in bins.keys()}
        for k in bins.keys():
            new_bin_values[k][1::2] += np.diff(bins[k])[0] / 2
        new_hist_values = {k: np.repeat(hists[k], 2) / 2 for k in bins.keys()}
        return new_bin_values, new_hist_values

    def spread_hists2(self, hists, factor):
        new_hist_values = {k: hists[k] / factor for k in hists.keys()}
        return new_hist_values

    def get_x_range(self, run_data):
        min = np.min([np.min(class_data) for class_name, class_data in run_data.items()])
        max = np.max([np.max(class_data) for class_name, class_data in run_data.items()])
        return min, max

    def get_all_hists_range(self, all_hists):
        x_min = np.min([np.min([np.min([np.min(all_hists[model_name][run_name]["bins"][trace]) for trace in
                                        self.trace_names]) for run_name in all_hists[model_name]]) for model_name in
                        all_hists])
        x_max = np.max([np.max([np.max([np.max(all_hists[model_name][run_name]["bins"][trace]) for trace in
                                        self.trace_names]) for run_name in all_hists[model_name]]) for model_name in
                        all_hists])
        y_min = np.min([np.min([np.min([np.min(all_hists[model_name][run_name]["hists"][trace]) for trace in
                                        self.trace_names]) for run_name in all_hists[model_name]]) for model_name in
                        all_hists])
        y_max = np.max([np.max([np.max([np.max(all_hists[model_name][run_name]["hists"][trace]) for trace in
                                        self.trace_names]) for run_name in all_hists[model_name]]) for model_name in
                        all_hists])
        return x_min, x_max, y_min, y_max

    def get_y_range(self, data):
        min_y = np.min([np.min([np.min([np.min(data[model_name][run_name][trace]) for trace in self.trace_names]) for
                                run_name in data[model_name]]) for model_name in data])
        max_y = np.max([np.max([np.max([np.max(data[model_name][run_name][trace]) for trace in self.trace_names]) for
                                run_name in data[model_name]]) for model_name in data])
        return min_y, max_y

    def main(self, names_to_keep: dict, type: str):
        all_json_data = self.collect_json(names_to_keep)
        if type == "hist":
            all_hists = self.make_histograms(all_json_data)
            self.plotting(all_hists, type="hist")
        elif type == "distplot":
            all_distplot = self.make_distplots(all_json_data)
            self.plotting(all_distplot, type="distplot")
        else:
            raise ValueError("type must be hist or distplot")


if __name__ == '__main__':
    names_to_keep_500 = {"TE": {"Best": 'results_idx_4', "Worst": 'results_idx_0'},
                         "Vanilla": {"Best": 'results_idx_3', "Worst": 'results_idx_0'},
                         "FAE": {"Best": 'results_idx_0', "Worst": 'results_idx_1'}}
    names_to_keep_50 = {"TE": {"Best": 'results_idx_3', "Worst": 'results_idx_1'},
                        "Vanilla": {"Best": 'results_idx_1', "Worst": 'results_idx_2'},
                        "FAE": {"Best": 'results_idx_1', "Worst": 'results_idx_0'}}
    names_to_keep = {50: names_to_keep_50, 500: names_to_keep_500}
    n_witnesses = 50

    subplot = Subplot_from_raw(n_bins=50, factor=2, n_witneses=n_witnesses)
    subplot.main(names_to_keep[n_witnesses], "distplot")
    print()
