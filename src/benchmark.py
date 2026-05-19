import os
import time
import argparse
import subprocess
import pandas as pd


def run_command(command):
    start = time.time()
    process = subprocess.run(command, shell=True)
    end = time.time()

    if process.returncode != 0:
        return None

    return end - start


def benchmark(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    tests = [
        {
            "method": "YOLOv8 + DeepSORT",
            "command": f'python src/yolo_deepsort.py --video "{video_path}" --output "{output_dir}/yolo_result.mp4"',
        },
        {
            "method": "Mask R-CNN + DeepSORT",
            "command": f'python src/maskrcnn_deepsort.py --video "{video_path}" --output "{output_dir}/maskrcnn_result.mp4"',
        },
    ]

    for test in tests:
        print(f"\nRunning: {test['method']}")

        elapsed_time = run_command(test["command"])

        if elapsed_time is None:
            results.append({
                "method": test["method"],
                "video": video_path,
                "status": "failed",
                "total_time_seconds": None,
                "output_video": None
            })
        else:
            results.append({
                "method": test["method"],
                "video": video_path,
                "status": "success",
                "total_time_seconds": round(elapsed_time, 2),
                "output_video": test["command"].split('--output "')[1].split('"')[0]
            })

    df = pd.DataFrame(results)

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    print("\nBenchmark finished.")
    print(df)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", default="outputs/benchmark")
    args = parser.parse_args()

    benchmark(args.video, args.output_dir)