"""Parse Slurm .log/.err pairs into per-run epoch_data.csv and final_metrics.csv."""

import os
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = "/n/fs/recbench/new_rec"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ERROR_DIR = RESULTS_DIR
FAILED_LOG_FILE = os.path.join(ERROR_DIR, "failed_operations.txt")

ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
metric_pat = re.compile(r"([a-z0-9@_]+)\s*[:=]\s*([\d\.]+)", re.IGNORECASE)
epoch_pat = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)
loss_pat = re.compile(r"train loss\s*[:=]\s*([\d\.]+)", re.IGNORECASE)


def find_log_by_id(target_dataset, target_model, folder_path):
    search_dirs = [os.path.join(BASE_DIR, "logs"), os.path.join(BASE_DIR, "logs_new"), BASE_DIR]
    all_files = []

    for d in search_dirs:
        if os.path.exists(d):
            all_files.extend(
                [os.path.join(d, f) for f in os.listdir(d) if f.endswith((".err", ".out", ".log"))]
            )

    all_files.sort(key=os.path.getmtime, reverse=True)

    search_str = f"THESIS_ID: {target_dataset}_{target_model}_{folder_path}"

    for path in all_files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i > 2000:
                        break

                    if search_str in line:
                        base, ext = os.path.splitext(path)
                        log_version = base + ".log"
                        err_version = base + ".err"

                        is_successful = False
                        if os.path.exists(log_version):
                            with open(
                                log_version, "r", encoding="utf-8", errors="ignore"
                            ) as check_f:
                                if "✅ Success" in check_f.read():
                                    is_successful = True

                        if not is_successful:
                            break

                        if os.path.exists(err_version):
                            print(
                                f"      Success verified; extracting data from {os.path.basename(err_version)}"
                            )
                            return err_version
                        else:
                            print(
                                f"      ⚠  Success verified, but .err missing. Extracting from {os.path.basename(path)}"
                            )
                            return path
        except Exception:
            continue

    return None


def parse_log_file(filepath, model_name):
    file_results = {}
    print(f"   Processing: {os.path.basename(filepath)}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    last_seen_epoch = -1
    debug_count = 0

    # tqdm writes \r to overwrite lines in a terminal; in a file they accumulate on one \n-line,
    # so split on both \n and \r to catch INFO lines embedded after tqdm bars
    for line in re.split(r"[\r\n]", raw):
        clean_line = ansi_escape.sub("", line).strip()

        epoch_match = re.search(r"epoch\s+(\d+)", clean_line, re.IGNORECASE)
        if epoch_match:
            last_seen_epoch = int(epoch_match.group(1))
            if debug_count < 3:
                print(f"      [DEBUG] Found Epoch {last_seen_epoch} on line: {clean_line}")
                debug_count += 1

            if last_seen_epoch not in file_results:
                file_results[last_seen_epoch] = {"Model": model_name, "Epoch": last_seen_epoch}

        if last_seen_epoch == -1:
            continue

        loss_match = re.search(r"train loss\s*[:=]\s*([\d\.]+)", clean_line, re.IGNORECASE)
        if loss_match:
            file_results[last_seen_epoch]["train_loss"] = float(loss_match.group(1))

        matches = re.findall(
            r"([a-z0-9@_]+)\s*[:=]\s*([-+]?\d*\.\d+|\d+)", clean_line, re.IGNORECASE
        )
        for key, val in matches:
            k_low = key.lower()
            if any(m in k_low for m in ["recall", "ndcg", "mrr", "precision", "hit", "valid"]):
                file_results[last_seen_epoch][k_low] = float(val)

    df = pd.DataFrame(list(file_results.values()))
    if not df.empty:
        print(f"      ✅ Successfully extracted {len(df)} epochs from {os.path.basename(filepath)}")
    else:
        print(
            "      ❌ ERROR: Regex failed to extract data. File might be empty or format is different."
        )
    return df


def _generate_plots(df_epochs, model_name, dataset_name, folder_path):
    plot_dir = os.path.join(folder_path, "epoch_plots")
    os.makedirs(plot_dir, exist_ok=True)
    valid_col = "valid_score" if "valid_score" in df_epochs.columns else None
    if "train_loss" in df_epochs.columns and valid_col:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_epochs, x="Epoch", y="train_loss", color="blue", label="Train Loss")
        ax2 = plt.twinx()
        sns.lineplot(
            data=df_epochs, x="Epoch", y=valid_col, color="orange", label="Validation", ax=ax2
        )
        plt.title(f"Convergence: {model_name} on {dataset_name}")
        plt.savefig(os.path.join(plot_dir, "training_convergence_dual.png"))
        plt.close()
    for m in [c for c in df_epochs.columns if c not in ["Model", "Epoch"]]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_epochs, x="Epoch", y=m, marker="o")
        plt.title(f"{model_name} ({dataset_name}) - {m.upper()}")
        clean_filename = m.replace("@", "_at_") + "_line.png"
        plt.savefig(os.path.join(plot_dir, clean_filename))
        plt.close()
    print(f"   Generated graphs in {plot_dir}")


print(f"Scanning {RESULTS_DIR} for experiments...")

failed_folders = []

for root, dirs, files in os.walk(RESULTS_DIR):
    folder_path = os.path.normpath(root)

    if "final_metrics.csv" in files:
        if "epoch_data.csv" in files:
            # epoch_data.csv already written by pipeline — just generate plots if missing
            plot_dir = os.path.join(folder_path, "epoch_plots")
            if not os.path.isdir(plot_dir):
                model_name = os.path.basename(folder_path)
                dataset_name = os.path.basename(os.path.dirname(folder_path))
                df_epochs = pd.read_csv(os.path.join(folder_path, "epoch_data.csv"))
                print(f"   Generating plots for: {dataset_name} -> {model_name}")
                _generate_plots(df_epochs, model_name, dataset_name, folder_path)
            else:
                print(
                    f"   Skipping: {os.path.basename(folder_path)} (epoch_data.csv + plots already exist)"
                )
            continue

        model_name = os.path.basename(folder_path)
        dataset_name = os.path.basename(os.path.dirname(folder_path))

        print(f"\nParsing new log for: {dataset_name} -> {model_name}")

        # RecBole's FileHandler writes plain-text logs at ./log/{model}/{model}-{dataset}-{time}-{md5}.log
        recbole_log_dir = os.path.join(BASE_DIR, "log", model_name)
        found_log = None
        if os.path.isdir(recbole_log_dir):
            candidates = [
                os.path.join(recbole_log_dir, f)
                for f in os.listdir(recbole_log_dir)
                if f.endswith(".log") and dataset_name in f
            ]
            if candidates:
                found_log = max(candidates, key=os.path.getmtime)
                print(f"      Found RecBole log: {os.path.basename(found_log)}")

        # fall back to Slurm .err files (older cluster behavior)
        if not found_log:
            found_log = find_log_by_id(dataset_name, model_name, folder_path)

        if found_log:
            df_epochs = parse_log_file(found_log, model_name)

            if not df_epochs.empty:
                df_epochs.to_csv(os.path.join(folder_path, "epoch_data.csv"), index=False)
                _generate_plots(df_epochs, model_name, dataset_name, folder_path)
            else:
                failed_folders.append(
                    f"{folder_path}  | Reason: Log matched, but no epoch data was extracted."
                )
        else:
            print(
                f"   ⚠  Could not find a successful log with matching THESIS_ID. = {dataset_name}_{model_name}_{folder_path}"
            )
            failed_folders.append(
                f"{folder_path}  | Reason: No successful log (.log + ✅ Success) found."
            )

print("\nAll structured results processed.")

if failed_folders:
    os.makedirs(ERROR_DIR, exist_ok=True)
    with open(FAILED_LOG_FILE, "w") as f:
        f.write("\n".join(failed_folders))
    print(
        f"NOTE: {len(failed_folders)} operations failed. A list has been saved to: {FAILED_LOG_FILE}"
    )
else:
    print("100% success — no failed operations were recorded.")
