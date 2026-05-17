# ============================================================
# EDS FINAL PROJECT - GRD-03: Grid Frequency Volatility
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/dataset_original.csv"
CLEANED_PATH = "data/dataset_cleaned.csv"
OUTPUT_PATH = "outputs/"

# ============================================================
# MODULE 1: DATA INGESTION
# ============================================================

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print("✔ Dataset loaded successfully!")
        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✘ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"✘ Unexpected error: {e}")
        return None

# ============================================================
# MODULE 2: DATA CLEANING
# ============================================================

def clean_data(df):
    """Clean dataset - handle nulls, duplicates, and data types."""
    try:
        print("\n--- Data Cleaning Report ---")
        
        # Check missing values
        missing = df.isnull().sum().sum()
        print(f"  Missing values found: {missing}")
        df = df.dropna()

        # Remove duplicates
        dupes = df.duplicated().sum()
        print(f"  Duplicate rows found: {dupes}")
        df = df.drop_duplicates()

        # Fix data types
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])

        # Map fault indicators to readable labels
        df['Fault Indicator'] = df['Fault Indicator'].map({
            0: 'No Fault',
            1: 'Overload',
            2: 'Short Circuit'
        })

        # Apply unique filter (GRD-03 unique slice)
        df = df[df['Fault Indicator'] != 'Short Circuit']
        print(f"  Unique filter applied: Excluded 'Short Circuit' faults")

        # Reset index
        df = df.reset_index(drop=True)

        print(f"  ✔ Cleaning complete! Remaining rows: {len(df)}")
        
        # Save cleaned dataset
        df.to_csv(CLEANED_PATH, index=False)
        print(f"  ✔ Cleaned dataset saved to {CLEANED_PATH}")
        
        return df
    except Exception as e:
        print(f"✘ Cleaning error: {e}")
        return None

# ============================================================
# MODULE 3: STATISTICAL ANALYSIS
# ============================================================

def analyze_data(df):
    """Compute descriptive and engineering statistics using NumPy."""
    try:
        print("\n--- Statistical Analysis ---")
        
        freq = df['Frequency (Hz)'].to_numpy()

        # Descriptive statistics
        mean   = np.mean(freq)
        median = np.median(freq)
        std    = np.std(freq)
        var    = np.var(freq)
        skew   = stats.skew(freq)
        kurt   = stats.kurtosis(freq)

        print(f"  Mean Frequency     : {mean:.4f} Hz")
        print(f"  Median Frequency   : {median:.4f} Hz")
        print(f"  Std Deviation      : {std:.4f} Hz")
        print(f"  Variance           : {var:.4f} Hz²")
        print(f"  Skewness           : {skew:.4f}")
        print(f"  Kurtosis           : {kurt:.4f}")

        # Outlier detection using IQR
        Q1 = np.percentile(freq, 25)
        Q3 = np.percentile(freq, 75)
        IQR = Q3 - Q1
        outliers = freq[(freq < Q1 - 1.5 * IQR) | (freq > Q3 + 1.5 * IQR)]
        print(f"  Outliers detected  : {len(outliers)}")

        # Correlation analysis
        voltage = df['Voltage (V)'].to_numpy()
        current = df['Current (A)'].to_numpy()
        power   = df['Power Usage (kW)'].to_numpy()

        corr_v = np.corrcoef(freq, voltage)[0, 1]
        corr_c = np.corrcoef(freq, current)[0, 1]
        corr_p = np.corrcoef(freq, power)[0, 1]

        print(f"\n  Correlation (Freq vs Voltage)    : {corr_v:.4f}")
        print(f"  Correlation (Freq vs Current)    : {corr_c:.4f}")
        print(f"  Correlation (Freq vs Power Usage): {corr_p:.4f}")

        # Comparative analysis: Fault vs No Fault
        fault    = df[df['Fault Indicator'] == 'Overload']['Frequency (Hz)'].to_numpy()
        no_fault = df[df['Fault Indicator'] == 'No Fault']['Frequency (Hz)'].to_numpy()

        print(f"\n  Avg Frequency (Overload)  : {np.mean(fault):.4f} Hz")
        print(f"  Avg Frequency (No Fault)  : {np.mean(no_fault):.4f} Hz")
        print(f"  Std Dev (Overload)        : {np.std(fault):.4f} Hz")
        print(f"  Std Dev (No Fault)        : {np.std(no_fault):.4f} Hz")

        # Inferential Statistics: Independent t-test
        t_stat, p_value = stats.ttest_ind(fault, no_fault)
        print(f"\n  Independent t-test (Overload vs No Fault):")
        print(f"  t-statistic : {t_stat:.4f}")
        print(f"  p-value     : {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Result: Significant difference (p < 0.05)")
        else:
            print(f"  Result: No significant difference (p ≥ 0.05)")

        print("\n  ✔ Analysis complete!")
        return mean, median, std, var, skew, kurt, outliers, corr_v, corr_c, corr_p, fault, no_fault, t_stat, p_value

    except Exception as e:
        print(f"✘ Analysis error: {e}")
        return None

# ============================================================
# MODULE 4: VISUALIZATION (STATIC GRAPHS)
# ============================================================

def visualize_static(df):
    """Generate static visualizations."""
    try:
        print("\n--- Generating Static Graphs ---")
        freq = df['Frequency (Hz)'].to_numpy()

        # --- Graph 1: Histogram of Frequency Distribution ---
        plt.figure(figsize=(10, 5))
        plt.hist(freq, bins=30, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(freq), color='red', linestyle='--', label=f'Mean: {np.mean(freq):.4f} Hz')
        plt.axvline(50.0, color='green', linestyle='--', label='Nominal: 50.0 Hz')
        plt.title('Grid Frequency Distribution')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'graph1_histogram.png')
        plt.close()
        print("  ✔ Graph 1 saved: Histogram")

        # --- Graph 2: Boxplot by Fault Type ---
        fault_groups = [
            df[df['Fault Indicator'] == 'No Fault']['Frequency (Hz)'].to_numpy(),
            df[df['Fault Indicator'] == 'Overload']['Frequency (Hz)'].to_numpy()
        ]
        plt.figure(figsize=(8, 6))
        plt.boxplot(fault_groups, labels=['No Fault', 'Overload'], patch_artist=True)
        plt.title('Frequency Distribution by Fault Type')
        plt.xlabel('Fault Type')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'graph2_boxplot.png')
        plt.close()
        print("  ✔ Graph 2 saved: Boxplot")

        # --- Graph 3: Scatter Plot - Frequency vs Voltage ---
        voltage = df['Voltage (V)'].to_numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(voltage, freq, alpha=0.4, color='purple', edgecolors='none')
        plt.title('Frequency vs Voltage')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH + 'graph3_scatter.png')
        plt.close()
        print("  ✔ Graph 3 saved: Scatter Plot")

        print("  ✔ All static graphs saved to outputs/")

    except Exception as e:
        print(f"✘ Visualization error: {e}")

# ============================================================
# MODULE 5: VISUALIZATION (ANIMATED GRAPHS)
# ============================================================

def visualize_animated(df):
    """Generate animated visualizations."""
    try:
        print("\n--- Generating Animated Graphs ---")

        # --- Animation 1: Frequency Over Time ---
        fig, ax = plt.subplots(figsize=(12, 5))
        freq_values = df['Frequency (Hz)'].to_numpy()
        time_index  = np.arange(len(freq_values))

        ax.set_xlim(0, len(freq_values))
        ax.set_ylim(freq_values.min() - 0.5, freq_values.max() + 0.5)
        ax.set_title('Grid Frequency Over Time')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Frequency (Hz)')
        ax.axhline(50.0, color='green', linestyle='--', linewidth=1, label='Nominal 50 Hz')
        ax.legend()

        line, = ax.plot([], [], color='steelblue', linewidth=1)

        def init1():
            line.set_data([], [])
            return line,

        def update1(frame):
            line.set_data(time_index[:frame], freq_values[:frame])
            return line,

        ani1 = animation.FuncAnimation(
            fig, update1, frames=len(freq_values),
            init_func=init1, interval=10, blit=True
        )
        ani1.save(OUTPUT_PATH + 'animation1_frequency_over_time.gif',
                  writer='pillow', fps=30)
        plt.close()
        print("  ✔ Animation 1 saved: Frequency Over Time")

        # --- Animation 2: Rolling Average Frequency ---
        fig, ax = plt.subplots(figsize=(12, 5))
        roll_avg = pd.Series(freq_values).rolling(window=20).mean().to_numpy()

        ax.set_xlim(0, len(freq_values))
        ax.set_ylim(freq_values.min() - 0.5, freq_values.max() + 0.5)
        ax.set_title('Rolling Average Grid Frequency (Window=20)')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Frequency (Hz)')
        ax.axhline(50.0, color='green', linestyle='--', linewidth=1, label='Nominal 50 Hz')
        ax.legend()

        line2, = ax.plot([], [], color='orange', linewidth=2)

        def init2():
            line2.set_data([], [])
            return line2,

        def update2(frame):
            line2.set_data(time_index[:frame], roll_avg[:frame])
            return line2,

        ani2 = animation.FuncAnimation(
            fig, update2, frames=len(freq_values),
            init_func=init2, interval=10, blit=True
        )
        ani2.save(OUTPUT_PATH + 'animation2_rolling_average.gif',
                  writer='pillow', fps=30)
        plt.close()
        print("  ✔ Animation 2 saved: Rolling Average Frequency")

        print("  ✔ All animations saved to outputs/")

    except Exception as e:
        print(f"✘ Animation error: {e}")

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = clean_data(df)
    results = analyze_data(df)
    visualize_static(df)
    visualize_animated(df)
