from wordcloud import WordCloud
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.metrics import f1_score, accuracy_score

def _format_peaklist_title(base):

    return base.split('_')[0]

def _format_testset_title(base):
    # Example: 'Suwannee_River_Fulvic_Acid_2' -> 'Suwannee River Fulvic Acid 2'
    mapping = {
        "Table_Pahokee_River_Fulvic_Acid": "PPFA",
        "Table_Suwannee_River_Fulvic_Acid_2_v2": "SRFA2",
        "Table_Suwannee_River_Fulvic_Acid_3": "SRFA3"
    }
    return mapping[base]

def _calculate_grid_dimensions(n_plots, max_cols=4):
    """Calculate optimal grid dimensions for n_plots with a maximum number of columns."""
    if n_plots <= max_cols:
        return 1, n_plots
    else:
        ncols = min(max_cols, n_plots) 
        nrows = math.ceil(n_plots / ncols) 
        return nrows, ncols


def _apply_subplot_spacing(fig, pad=2.0, wspace=0.45, hspace=0.55):
    """Apply consistent spacing to subplot figures.

    Args:
        fig: matplotlib Figure
        pad: padding for tight_layout
        wspace: width space between subplots (subplots_adjust)
        hspace: height space between subplots (subplots_adjust)
    """
    try:
        fig.tight_layout(pad=pad)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    except Exception as e:
        plot_logger.warning(f"Could not adjust subplot spacing: {e}")



def plot_wordcloud_grid(result_dirs, labels, out_dir, is_peaklist=False, top_n=20):
    # Find all unique files (by base name) across all result_dirs
    base_files = set()
    for result_dir in result_dirs:
        sub_dir = os.path.join(result_dir, 'peak_list') if is_peaklist else result_dir
        if os.path.exists(sub_dir):
            for file in os.listdir(sub_dir):
                if (is_peaklist and file.endswith('.csv')) or (not is_peaklist and file.startswith('results_') and file.endswith('.csv')):
                    base = file.replace('results_', '').replace('.csv', '')
                    base_files.add(base)
    
    # Filter peaklist files to only include specified ones
    if is_peaklist:
        specified_files = ['PPFA_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec', 
                          'SRFA2_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec', 
                          'SRFA3_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec']
        base_files = [f for f in base_files if f in specified_files]
    
    base_files = sorted(list(base_files))
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if is_peaklist:
        # Create separate plots for each peaklist file
        for base in base_files:
            # Most common formulas
            n_labels = len(labels)
            nrows, ncols = _calculate_grid_dimensions(n_labels, max_cols=4)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            
            # Handle different axis configurations
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                sub_dir = os.path.join(result_dir, 'peak_list')
                file = f'{base}.csv'
                ax = axes[idx]
                path = os.path.join(sub_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = df[df['valid_prediction'] == 1]
                    if not df.empty and 'predicted_formula' in df.columns:
                        freq = df['predicted_formula'].value_counts().head(top_n)
                        if len(freq) > 0:
                            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(freq)
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{label} - {_format_peaklist_title(base)}')
                        else:
                            ax.axis('off')
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.axis('off')
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"peaklist_wordcloud_common_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Peaklist most common wordcloud plot saved: {out_path}")

            # Most unique formulas (appearing only once)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            
            # Handle different axis configurations
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                sub_dir = os.path.join(result_dir, 'peak_list')
                file = f'{base}.csv'
                ax = axes[idx]
                path = os.path.join(sub_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = df[df['valid_prediction'] == 1]
                    if not df.empty and 'predicted_formula' in df.columns:
                        unique = df['predicted_formula'].value_counts()
                        unique = unique[unique == 1].head(top_n)
                        if len(unique) > 0:
                            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(unique)
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{label} - {_format_peaklist_title(base)}')
                        else:
                            ax.axis('off')
                            ax.text(0.5, 0.5, 'No unique data', ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.axis('off')
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"peaklist_wordcloud_unique_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Peaklist unique wordcloud plot saved: {out_path}")
    else:
        # Test results: create separate plots for each test set
        for base in base_files:
            n_labels = len(labels)
            nrows, ncols = _calculate_grid_dimensions(n_labels, max_cols=4)
            
            # Most common formulas
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            
            # Handle different axis configurations
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                sub_dir = result_dir
                file = f'results_{base}.csv'
                ax = axes[idx]
                path = os.path.join(sub_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = df[(df['true_prediction'] == 1) | (df['false_prediction'] == 1)]
                    if not df.empty and 'predicted_formula' in df.columns:
                        freq = df['predicted_formula'].value_counts().head(top_n)
                        if len(freq) > 0:
                            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(freq)
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{label} {_format_testset_title(base)}')
                        else:
                            ax.axis('off')
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.axis('off')
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"test_results_wordcloud_common_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Test results most common wordcloud plot saved: {out_path}")

            # Most unique formulas (appearing only once)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            
            # Handle different axis configurations
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                sub_dir = result_dir
                file = f'results_{base}.csv'
                ax = axes[idx]
                path = os.path.join(sub_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df = df[(df['true_prediction'] == 1) | (df['false_prediction'] == 1)]
                    if not df.empty and 'predicted_formula' in df.columns:
                        unique = df['predicted_formula'].value_counts()
                        unique = unique[unique == 1].head(top_n)
                        if len(unique) > 0:
                            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(unique)
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'{label} - {_format_testset_title(base)}')
                        else:
                            ax.axis('off')
                            ax.text(0.5, 0.5, 'No unique data', ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.axis('off')
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"test_results_wordcloud_unique_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Test results unique wordcloud plot saved: {out_path}")

plt.rcParams.update({
    'font.size':60,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 1.0,
})

def get_plot_logger():
    """Set up and return a logger for plotting."""
    PLOT_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'plotting.log')
    os.makedirs(os.path.dirname(PLOT_LOG_FILE), exist_ok=True)
    logger = logging.getLogger("plotting")
    if not logger.handlers:
        fh = logging.FileHandler(PLOT_LOG_FILE)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

plot_logger = get_plot_logger()


def plot_roc_auc(y_true, y_score, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    plot_logger.info(f"ROC AUC curve saved: {out_path}")


def plot_test_results_grid(result_dirs, labels, out_dir):
    # Find all unique result files (by base name) across all result_dirs
    base_files = set()
    for result_dir in result_dirs:
        for file in os.listdir(result_dir):
            if file.startswith('results_') and file.endswith('.csv'):
                base = file.replace('results_', '').replace('.csv', '')
                base_files.add(base)
    base_files = sorted(list(base_files))
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Create separate plots for each test set
    for base in base_files:
        print(f"Plotting test results for: {base}")
        n_labels = len(labels)
        nrows, ncols = _calculate_grid_dimensions(n_labels, max_cols=4)
        
        # Bar plots (true/false/no) - separate plot for each test set
        fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols, 10*nrows), sharey=True)
        
        # Handle different axis configurations
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        try:    
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                file = f'results_{base}.csv'
                ax = axes[idx]
                path = os.path.join(result_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    counts = [
                        df['true_prediction'].sum() if 'true_prediction' in df.columns else 0,
                        df['false_prediction'].sum() if 'false_prediction' in df.columns else 0,
                        # df['no_prediction'].sum() if 'no_prediction' in df.columns else 0,
                    ]
                    categories = ['Predicted \nFormula', 'New Formula \n Assignments']
                    sns.barplot(x=categories, y=counts, palette='Set2', ax=ax)
                    for j, v in enumerate(counts):
                        ax.text(j, v + max(counts) * 0.01 if max(counts) > 0 else 0.1, str(int(v)), ha='center', va='bottom', fontweight='bold')
                else:
                    # Show empty plot with message if file doesn't exist
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
                    
                ax.set_title(f'{label} - {_format_testset_title(base)}')
                ax.set_ylabel('Count')
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"test_results_prediction_types_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Test results prediction types plot saved: {out_path}")

            # Distribution plots (mass error) - separate plot for each test set
            fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), sharey=True)
            
            # Handle different axis configurations
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes
            else:
                axes = axes.flatten()
                
            for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
                file = f'results_{base}.csv'
                ax = axes[idx]
                path = os.path.join(result_dir, file)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'mass_error_in_ppm' in df.columns:
                        filtered = df[df['mass_error_in_ppm'] <= 1]
                        if not filtered.empty:
                            sns.histplot(filtered['mass_error_in_ppm'], bins=50, kde=True, ax=ax)
                        else:
                            ax.text(0.5, 0.5, 'No data <= 1ppm', ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, 'No mass error data', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'File not found', ha='center', va='center', transform=ax.transAxes)
                    
                ax.set_title(f'{label} - {_format_testset_title(base)}')
                ax.set_ylabel('Count')
                ax.set_xlabel('Mass Error (ppm)')
            
            # Hide unused subplots
            for idx in range(len(labels), nrows * ncols):
                axes[idx].axis('off')
            
            plt.tight_layout()
            out_path = os.path.join(plots_dir, f"test_results_mass_error_{base}.png")
            plt.savefig(out_path)
            plt.close()
            plot_logger.info(f"Test results mass error plot saved: {out_path}")
        except Exception as e:
                print(f"Error plotting test results for {base}: {e}")


def plot_test_results_grid_combined(result_dirs, labels, out_dir):
    """Aggregate all result files per experiment (result_dir) and plot summary grids.

    For each experiment (approach) we aggregate counts across every results_*.csv
    file inside that directory and then:
      1. Create a grid of bar charts (one subplot per experiment) showing
         total Predicted Formula (true_prediction) vs New Formula Assignments (false_prediction).
      2. Create a grid of mass error histograms (<=1 ppm) aggregated across all files.

            Additionally computes per-experiment classification metrics treating:
            class 1 = predicted (true_prediction OR false_prediction == 1)
            class 0 = no prediction (no_prediction == 1)

        Where provided CSVs lack an explicit no_prediction column we infer it from
        (1 - max(true_prediction,false_prediction)) if a 'no_prediction' column isn't present.

            Metrics saved:
                - metrics_test_results_combined.csv (accuracy, f1_score, support counts)
                - test_results_confusion_matrices_combined.png (all confusion matrices grid)

        Saved figures:
            - test_results_prediction_types_combined.png (bar grid)
            - test_results_mass_error_combined.png (mass error grid)
    """
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    metrics_rows = []  # Collect metrics per experiment
    confusion_matrices = []  # store (label, confusion_matrix)

    try:
        n_labels = len(labels)
        nrows, ncols = _calculate_grid_dimensions(n_labels, max_cols=4)

        # ------------------ Bar Plot Grid ------------------
        fig, axes = plt.subplots(nrows, ncols, figsize=(16*ncols, 12*nrows), sharey=True)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
            ax = axes[idx]
            true_total = 0
            false_total = 0
            no_total = 0
            # y_true_all/y_pred_all not required for metrics (constructed via aggregate counts)

            if os.path.isdir(result_dir):
                for file in os.listdir(result_dir):
                    if file.startswith('results_') and file.endswith('.csv'):
                        path = os.path.join(result_dir, file)
                        try:
                            df = pd.read_csv(path)
                            # Summations
                            t_sum = df['true_prediction'].sum() if 'true_prediction' in df.columns else 0
                            f_sum = df['false_prediction'].sum() if 'false_prediction' in df.columns else 0
                            # no_prediction handling
                            if 'no_prediction' in df.columns:
                                n_sum = df['no_prediction'].sum()
                            else:
                                # infer: a row with both true and false 0 -> no
                                if 'true_prediction' in df.columns and 'false_prediction' in df.columns:
                                    n_sum = ((df['true_prediction'] == 0) & (df['false_prediction'] == 0)).sum()
                                else:
                                    n_sum = 0
                            true_total += t_sum
                            false_total += f_sum
                            no_total += n_sum

                            # Construct y_true/y_pred: treat any prediction (true/false) as class 1 ground truth
                            # Row-level presence flags can be derived if needed for future metrics
                            # (kept minimal now to avoid unused variable warnings)
                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error reading {file}', ha='center', va='center', transform=ax.transAxes)
                            plot_logger.error(f"Error reading {path}: {e}")
            else:
                ax.text(0.5, 0.5, 'Dir not found', ha='center', va='center', transform=ax.transAxes)

            counts = [true_total, false_total]
            categories = ['Predicted \nFormula', 'New Formula \nAssignments']
            sns.barplot(x=categories, y=counts, palette='Set2', ax=ax)
            for j, v in enumerate(counts):
                ax.text(j, v * (0.5) if v > 10 else v, str(int(v)), ha='center', va='bottom', fontweight='bold')
            ax.set_title(label)
            ax.set_ylabel('Count')

            # Metrics
            total_samples = true_total + false_total + no_total
            if total_samples > 0:
                # Build confusion matrix: class 1 predictions = true_total + false_total; class 0 = no_total
                # Because y_pred == y_true (assignment presence) f1 is 1 if any predictions else 0.
                y_true_vec = [1]* (true_total + false_total) + [0]* no_total
                y_pred_vec = [1]* (true_total + false_total) + [0]* no_total
                precision = true_total / (true_total + false_total) if (true_total + false_total) > 0 else 0.0
                assignment_rate = (true_total + false_total) / total_samples if total_samples > 0 else 0.0
                assignment_rate = round(assignment_rate* 100, 3)
                cm = confusion_matrix(y_true_vec, y_pred_vec, labels=[0,1])
                metrics_rows.append({
                    'label': label,
                    'True Predictions': true_total,
                    'New Formula Assignments': false_total,
                    'False Predictions': no_total,
                    'Total Samples': total_samples,
                    # 'Precision Within Assigned': precision,
                    'Assignment Rate': assignment_rate
                })

                confusion_matrices.append((label, cm))

        # Hide unused subplots
        for idx in range(len(labels), nrows * ncols):
            axes[idx].axis('off')

        _apply_subplot_spacing(fig, pad=0.5, wspace=0.2, hspace=0.6)
        out_path = os.path.join(plots_dir, 'test_results_prediction_types_combined.png')
        plt.savefig(out_path)
        plt.close()
        plot_logger.info(f"Combined test results prediction types plot saved: {out_path}")

        # ------------------ Confusion Matrix Grid ------------------
        if confusion_matrices:
            fig_cm_grid, axes_cm = plt.subplots(nrows, ncols, figsize=(16*ncols, 12*nrows))
            if nrows == 1 and ncols == 1:
                axes_cm = [axes_cm]
            elif nrows == 1:
                axes_cm = axes_cm
            else:
                axes_cm = axes_cm.flatten()
            for idx, (label, cm) in enumerate(confusion_matrices):
                ax_cm = axes_cm[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                            xticklabels=['False','True'], yticklabels=['False','True'], ax=ax_cm)
                ax_cm.set_title(label)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Ground Truth')
            # Hide unused
            for idx in range(len(confusion_matrices), nrows * ncols):
                axes_cm[idx].axis('off')
            _apply_subplot_spacing(fig_cm_grid, pad=0.2, wspace=0.2, hspace=0.4)
            cm_grid_path = os.path.join(plots_dir, 'test_results_confusion_matrices_combined.png')
            plt.savefig(cm_grid_path)
            plt.close()
            plot_logger.info(f"Combined confusion matrices plot saved: {cm_grid_path}")

        # ------------------ Mass Error Histogram Grid ------------------
        fig, axes = plt.subplots(nrows, ncols, figsize=(16*ncols, 12*nrows), sharey=True)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
            ax = axes[idx]
            mass_errors = []
            if os.path.isdir(result_dir):
                for file in os.listdir(result_dir):
                    if file.startswith('results_') and file.endswith('.csv'):
                        path = os.path.join(result_dir, file)
                        try:
                            df = pd.read_csv(path)
                            if 'mass_error_in_ppm' in df.columns:
                                filtered = df[df['mass_error_in_ppm'] <= 1]['mass_error_in_ppm'].dropna().tolist()
                                mass_errors.extend(filtered)
                        except Exception as e:
                            plot_logger.error(f"Error reading {path}: {e}")
            if mass_errors:
                sns.histplot(mass_errors, bins=50, kde=True, ax=ax)
                ax.set_xlabel('Mass Error (ppm)')
            else:
                ax.text(0.5, 0.5, 'No data <= 1ppm', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            ax.set_ylabel('Count')

        for idx in range(len(labels), nrows * ncols):
            axes[idx].axis('off')

        _apply_subplot_spacing(fig, pad=0.5, wspace=0.2, hspace=0.4)
        out_path = os.path.join(plots_dir, 'test_results_mass_error_combined.png')
        plt.savefig(out_path)
        plt.close()
        plot_logger.info(f"Combined test results mass error plot saved: {out_path}")

    except Exception as e:
        plot_logger.error(f"Error generating combined test results grid: {e}")
    # Save metrics CSV
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(plots_dir, 'metrics_test_results_combined.csv')
        metrics_df.to_csv(metrics_path, index=False)
        plot_logger.info(f"Combined test results metrics saved: {metrics_path}")


def plot_test_results_wordcloud_combined(result_dirs, labels, out_dir, top_n=30):
    """Create combined wordclouds (common & unique) across ALL test set result files per experiment.

    For each experiment (result_dir):
      - Aggregate all `predicted_formula` values from every results_*.csv where either
        true_prediction==1 or false_prediction==1 (assigned formulas).
      - Build frequency counts and generate a wordcloud of the top_n most common formulas.
      - Identify formulas appearing exactly once (unique across that experiment) and generate a unique wordcloud.

    Two grid figures are saved (one subplot per experiment):
      - test_results_wordcloud_combined_common.png
      - test_results_wordcloud_combined_unique.png
    """
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if not result_dirs:
        plot_logger.warning("No result directories provided for combined test wordclouds.")
        return

    n_labels = len(labels)
    nrows, ncols = _calculate_grid_dimensions(n_labels, max_cols=4)

    # ------------------ Common (Top-N) Wordclouds ------------------
    fig_common, axes_common = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.2*nrows))
    if nrows == 1 and ncols == 1:
        axes_common = [axes_common]
    elif nrows == 1:
        axes_common = axes_common
    else:
        axes_common = axes_common.flatten()

    for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
        ax = axes_common[idx]
        all_formulas = []
        if os.path.isdir(result_dir):
            for file in os.listdir(result_dir):
                if file.startswith('results_') and file.endswith('.csv'):
                    path = os.path.join(result_dir, file)
                    try:
                        df = pd.read_csv(path)
                        if {'predicted_formula','true_prediction','false_prediction'}.issubset(df.columns):
                            assigned = df[(df['true_prediction'] == 1) | (df['false_prediction'] == 1)]
                            if not assigned.empty:
                                all_formulas.extend(assigned['predicted_formula'].dropna().tolist())
                    except Exception as e:
                        plot_logger.error(f"Error reading {path}: {e}")
                        ax.text(0.5,0.5,'Read error',ha='center',va='center',transform=ax.transAxes)
        if all_formulas:
            freq = pd.Series(all_formulas).value_counts().head(5)
            if len(freq) > 0:
                print(freq)
                wc = WordCloud(width=500, height=350, background_color='white').generate_from_frequencies(freq)
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(label)
            else:
                ax.axis('off')
                ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        else:
            ax.axis('off')
            ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)

    for idx in range(len(labels), nrows * ncols):
        axes_common[idx].axis('off')
    _apply_subplot_spacing(fig_common, pad=0.2, wspace=0.2, hspace=0.4)
    common_path = os.path.join(plots_dir, 'test_results_wordcloud_combined_common.png')
    plt.savefig(common_path)
    plt.close(fig_common)
    plot_logger.info(f"Combined test common wordclouds saved: {common_path}")

    # ------------------ Unique (Appearing Once) Wordclouds ------------------
    fig_unique, axes_unique = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.2*nrows))
    if nrows == 1 and ncols == 1:
        axes_unique = [axes_unique]
    elif nrows == 1:
        axes_unique = axes_unique
    else:
        axes_unique = axes_unique.flatten()

    for idx, (result_dir, label) in enumerate(zip(result_dirs, labels)):
        ax = axes_unique[idx]
        all_formulas = []
        if os.path.isdir(result_dir):
            for file in os.listdir(result_dir):
                if file.startswith('results_') and file.endswith('.csv'):
                    path = os.path.join(result_dir, file)
                    try:
                        df = pd.read_csv(path)
                        if {'predicted_formula','true_prediction','false_prediction'}.issubset(df.columns):
                            assigned = df[(df['true_prediction'] == 1) | (df['false_prediction'] == 1)]
                            if not assigned.empty:
                                all_formulas.extend(assigned['predicted_formula'].dropna().tolist())
                    except Exception as e:
                        plot_logger.error(f"Error reading {path}: {e}")
                        ax.text(0.5,0.5,'Read error',ha='center',va='center',transform=ax.transAxes)
        if all_formulas:
            vc = pd.Series(all_formulas).value_counts()
            unique = vc[vc == 1].head(top_n)
            if len(unique) > 0:
                wc = WordCloud(width=500, height=350, background_color='white').generate_from_frequencies(unique)
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(label)
            else:
                ax.axis('off')
                ax.text(0.5,0.5,'No unique',ha='center',va='center',transform=ax.transAxes)
        else:
            ax.axis('off')
            ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)

    for idx in range(len(labels), nrows * ncols):
        axes_unique[idx].axis('off')
    _apply_subplot_spacing(fig_unique, pad=1.5, wspace=0.3, hspace=0.45)
    unique_path = os.path.join(plots_dir, 'test_results_wordcloud_combined_unique.png')
    plt.savefig(unique_path)
    plt.close(fig_unique)
    plot_logger.info(f"Combined test unique wordclouds saved: {unique_path}")


def group_peaklist_files_by_sample(base_files):
    # Group base_files by sample type prefix (e.g. PPFA, SRFA2, SRFA3)
    groups = {'PPFA': [], 'SRFA2': [], 'SRFA3': []}
    for base in base_files:
        if base.startswith('PPFA'):
            groups['PPFA'].append(base)
        elif base.startswith('SRFA2'):
            groups['SRFA2'].append(base)
        elif base.startswith('SRFA3'):
            groups['SRFA3'].append(base)
    # Only keep non-empty groups, preserve order
    ordered = [groups[k] for k in ['PPFA', 'SRFA2', 'SRFA3'] if groups[k]]
    return ordered


def plot_peaklist_grid(result_dirs, labels, out_dir):
    # Use only specified peaklist files
    specified_files = ['PPFA_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec', 
                      'SRFA2_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec', 
                      'SRFA3_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec']
    
    # Composer counts for each peaklist
    composer_counts = {
        'PPFA_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec': 2213,
        'SRFA2_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec': 1968,
        'SRFA3_neg_8M_0.5s_5ppm_aFTk_PeakList_Rec': 1733
    } ##Provided by Composer analysis of the same peaklists
    
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # We'll collect a combined CSV summary per peaklist/approach
    combined_rows = []

    # Create a single figure with 3 rows (one for each peaklist file)
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))

    # Process each peaklist file in its own subplot row
    for idx, base in enumerate(specified_files):
        ax = axes[idx]
        
        # Collect data for all approaches for this file
        valid_counts = []
        approach_labels = []
        
        for result_dir, label in zip(result_dirs, labels):
            peaklist_dir = os.path.join(result_dir, 'peak_list')
            file = f'{base}.csv'
            path = os.path.join(peaklist_dir, file)
            
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    plot_logger.error(f"Error reading {path}: {e}")
                    df = pd.DataFrame()
                valid = int(df['valid_prediction'].sum()) if 'valid_prediction' in df.columns and not df.empty else 0
                total_rows = int(len(df)) if not df.empty else 0
                file_exists = True
                valid_counts.append(valid)
                approach_labels.append(label)
            else:
                valid = 0
                total_rows = 0
                file_exists = False
                valid_counts.append(0)
                approach_labels.append(f"{label} (missing)")

            # Append row for combined CSV
            combined_rows.append({
                'peaklist': base,
                'approach': label,
                'file_exists': file_exists,
                'total_rows': total_rows,
                'valid_predictions': valid,
            })

        
        # Add Composer bar as the first bar
        composer_count = composer_counts.get(base, 0)
        all_labels = ['Composer'] + approach_labels
        all_counts = [composer_count] + valid_counts
        
        colors_list = ['#FF6B6B'] + [plt.cm.tab20(i) for i in range(len(approach_labels))]
        bars = ax.bar(all_labels, all_counts, color=colors_list)
        
        # Add value labels on bars
        for bar, count in zip(bars, all_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(all_counts) * 0.01 if max(all_counts) > 0 else 0.1,
                   str(int(count)), ha='center', va='bottom', fontweight='bold', fontsize=18)
        
        # Set y-axis limit with padding at the top to prevent label overflow
        if max(all_counts) > 0:
            ax.set_ylim(0, max(all_counts) * 1.15)
        
        ax.set_title(f'Formula Assigned - {_format_peaklist_title(base)}', fontsize=18, fontweight='bold')
        ax.set_ylabel('Formula Count', fontsize=13)
        
        # Only show x-axis labels on the last row
        if idx < len(specified_files) - 1:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
        else:
            ax.set_xlabel('Approach', fontsize=13)
            ax.tick_params(axis='x', rotation=45, labelsize=18)
        
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout and save
    _apply_subplot_spacing(fig, pad=0.5, wspace=0.2, hspace=0.2)
    out_path = os.path.join(plots_dir, f"peaklist_valid_prediction_combined.png")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    plot_logger.info(f"Combined peaklist valid prediction plot saved: {out_path}")

    # Save combined summary CSV for peaklists
    try:
        if combined_rows:
            combined_df = pd.DataFrame(combined_rows)
            csv_path = os.path.join(plots_dir, 'peaklist_combined_summary.csv')
            combined_df.to_csv(csv_path, index=False)
            plot_logger.info(f"Combined peaklist summary CSV saved: {csv_path}")
    except Exception as e:
        plot_logger.error(f"Error saving combined peaklist summary CSV: {e}")


def plot_peaklist_main(result_dirs, labels, out_dir):
    """Main function to generate all peaklist plots."""
    plot_logger.info("Starting peaklist plotting...")
    plot_peaklist_grid(result_dirs, labels, out_dir)
    plot_wordcloud_grid(result_dirs, labels, out_dir, is_peaklist=True)
    plot_logger.info("Finished peaklist plotting.")

def plot_testset_main(result_dirs, labels, out_dir):
    return
    """Main function to generate all test set plots."""
    plot_logger.info("Starting test set plotting...")
    # plot_test_results_grid(result_dirs, labels, out_dir)
    # plot_wordcloud_grid(result_dirs, labels, out_dir, is_peaklist=False)
    plot_test_results_grid_combined(result_dirs, labels, out_dir)
    plot_test_results_wordcloud_combined(result_dirs, labels, out_dir)
    plot_logger.info("Finished test set plotting.")
