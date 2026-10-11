# main.py — single entry point
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Nuclear Power Plant Anomaly Detection Benchmark Pipeline")
    parser.add_argument('--quick', action='store_true', help='Use 5 epochs and few scenarios for fast testing')
    args = parser.parse_args()

    # Pass QUICK mode via environment variable and config
    if args.quick:
        os.environ['NPPAD_QUICK'] = '1'
        print("[INFO] Quick mode enabled: 5 epochs, max 10 scenarios per accident type.")
    else:
        os.environ['NPPAD_QUICK'] = '0'
        print("[INFO] Full mode enabled: 30 epochs, all scenarios.")

    steps = ['explore_data.py', 'preprocess.py', 'models_baseline.py', 'models_proposed.py', 'evaluate.py']
    for step in steps:
        print(f"\n{'='*75}")
        print(f"=== RUNNING: {step} ===")
        print(f"{'='*75}")
        result = subprocess.run([sys.executable, step], capture_output=False)
        if result.returncode != 0:
            print(f"\n[ERROR] Pipeline step {step} failed with return code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'='*75}")
    print("[SUCCESS] All pipeline steps completed successfully! Results are saved in 'results/'.")
    print(f"{'='*75}")

if __name__ == '__main__':
    main()
