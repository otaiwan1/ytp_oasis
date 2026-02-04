#!/usr/bin/env python3
"""
OASIS Feedback Pipeline - 牙醫師回饋與模型優化工具

這個腳本整合了整個 fine-tuning 流程，讓牙醫師可以一鍵完成：
1. 收集回饋 (Collect Feedback) - 評分模型找到的相似配對
2. 微調模型 (Fine-tune) - 根據回饋優化模型
3. 驗證效果 (Validate) - 比較微調前後的效果

Usage:
    python oasis_feedback_pipeline.py          # 執行完整流程
    python oasis_feedback_pipeline.py --skip-feedback  # 跳過回饋收集（用於測試）
    python oasis_feedback_pipeline.py --status # 查看目前狀態
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# --- 路徑設定 ---
SCRIPT_DIR = Path(__file__).parent.resolve()
FINETUNE_DIR = SCRIPT_DIR / "fine-tuning"
VALIDATION_DIR = SCRIPT_DIR / "validation"

COLLECT_SCRIPT = FINETUNE_DIR / "collect_feedback.py"
FINETUNE_SCRIPT = FINETUNE_DIR / "finetune_feedback.py"
VALIDATE_SCRIPT = VALIDATION_DIR / "validate_model.py"

# History directories
MODEL_HISTORY_DIR = FINETUNE_DIR / "model_history"
FEEDBACK_HISTORY_DIR = FINETUNE_DIR / "feedback_history"
LATEST_MODEL_INFO = MODEL_HISTORY_DIR / "latest.txt"


def print_header(text):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(step_num, total, description):
    """Print step indicator."""
    print(f"\n{'─' * 60}")
    print(f"  📍 步驟 {step_num}/{total}: {description}")
    print(f"{'─' * 60}\n")


def get_latest_model():
    """Get the latest fine-tuned model path."""
    if LATEST_MODEL_INFO.exists():
        with open(LATEST_MODEL_INFO, 'r') as f:
            path = Path(f.read().strip())
            if path.exists():
                return path
    return None


def get_latest_feedback():
    """Get the most recent feedback file from history."""
    if not FEEDBACK_HISTORY_DIR.exists():
        return None
    
    files = sorted(FEEDBACK_HISTORY_DIR.glob("feedback_*.csv"))
    return files[-1] if files else None


def count_models():
    """Count fine-tuned models."""
    if not MODEL_HISTORY_DIR.exists():
        return 0
    return len(list(MODEL_HISTORY_DIR.glob("v*_*.pth")))


def count_feedback_files():
    """Count feedback history files."""
    if not FEEDBACK_HISTORY_DIR.exists():
        return 0
    return len(list(FEEDBACK_HISTORY_DIR.glob("*.csv")))


def show_status():
    """Show current system status."""
    print_header("OASIS 系統狀態")
    
    # Model status
    latest_model = get_latest_model()
    model_count = count_models()
    
    print(f"\n📦 模型狀態:")
    print(f"   • Fine-tuned 模型數量: {model_count}")
    if latest_model:
        print(f"   • 目前使用模型: {latest_model.name}")
    else:
        print(f"   • 目前使用模型: Base Model (尚未 fine-tune)")
    
    # Feedback status
    feedback_count = count_feedback_files()
    latest_feedback = get_latest_feedback()
    
    print(f"\n📋 回饋狀態:")
    print(f"   • 回饋記錄數量: {feedback_count}")
    if latest_feedback:
        print(f"   • 最新回饋: {latest_feedback.name}")
    
    # Check feedback_report.csv
    main_feedback = FINETUNE_DIR / "feedback_report.csv"
    if main_feedback.exists():
        import pandas as pd
        df = pd.read_csv(main_feedback)
        print(f"   • 累積回饋筆數: {len(df)}")
    
    print()


def run_collect_feedback():
    """Run the feedback collection tool."""
    print("🖥️  啟動回饋收集工具...")
    print("   請在彈出的視窗中評分相似度 (1-10分)")
    print("   完成後關閉視窗即可繼續\n")
    
    result = subprocess.run(
        [sys.executable, str(COLLECT_SCRIPT)],
        cwd=str(FINETUNE_DIR)
    )
    
    if result.returncode != 0:
        print("⚠️  回饋收集可能未正常完成")
        return False
    
    return True


def run_finetune(feedback_file=None):
    """Run fine-tuning with the latest feedback."""
    cmd = [sys.executable, str(FINETUNE_SCRIPT)]
    
    if feedback_file:
        cmd.extend(["--report", str(feedback_file)])
    
    result = subprocess.run(cmd, cwd=str(FINETUNE_DIR))
    
    return result.returncode == 0


def run_validation():
    """Run validation comparing base vs latest fine-tuned model."""
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--compare"],
        cwd=str(VALIDATION_DIR)
    )
    
    return result.returncode == 0


def run_pipeline(skip_feedback=False):
    """Run the complete feedback pipeline."""
    start_time = datetime.now()
    
    print_header("🦷 OASIS 牙齒相似度模型優化流程")
    print(f"\n開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show current status
    show_status()
    
    # Record state before
    models_before = count_models()
    
    # --- Step 1: Collect Feedback ---
    if not skip_feedback:
        print_step(1, 3, "收集醫師回饋")
        
        success = run_collect_feedback()
        if not success:
            response = input("\n是否繼續進行 fine-tuning? (y/n): ").strip().lower()
            if response != 'y':
                print("流程已取消。")
                return
    else:
        print_step(1, 3, "收集醫師回饋 (已跳過)")
        print("   使用現有的回饋資料進行 fine-tuning")
    
    # --- Step 2: Fine-tune ---
    print_step(2, 3, "微調模型")
    
    # Get the latest feedback file
    latest_feedback = get_latest_feedback()
    if latest_feedback and not skip_feedback:
        print(f"   使用最新回饋: {latest_feedback.name}")
        success = run_finetune(feedback_file=latest_feedback)
    else:
        print(f"   使用所有累積回饋")
        success = run_finetune()
    
    if not success:
        print("❌ Fine-tuning 失敗")
        return
    
    # Check if new model was created
    models_after = count_models()
    if models_after > models_before:
        latest_model = get_latest_model()
        print(f"\n✅ 新模型已建立: {latest_model.name}")
    
    # --- Step 3: Validate ---
    print_step(3, 3, "驗證模型效果")
    
    success = run_validation()
    
    # --- Summary ---
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("🎉 流程完成!")
    print(f"\n⏱️  總耗時: {duration.seconds // 60} 分 {duration.seconds % 60} 秒")
    print(f"📦 目前模型版本數: {count_models()}")
    
    latest_model = get_latest_model()
    if latest_model:
        print(f"🏆 最新模型: {latest_model.name}")
    
    print("\n下次執行此腳本將使用新模型繼續優化！")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OASIS 牙醫師回饋與模型優化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python oasis_feedback_pipeline.py              # 執行完整流程
  python oasis_feedback_pipeline.py --skip-feedback  # 跳過回饋收集
  python oasis_feedback_pipeline.py --status     # 查看系統狀態
        """
    )
    
    parser.add_argument(
        "--skip-feedback", 
        action="store_true",
        help="跳過回饋收集步驟，直接使用現有回饋進行 fine-tuning"
    )
    
    parser.add_argument(
        "--status",
        action="store_true", 
        help="只顯示目前系統狀態，不執行流程"
    )
    
    args = parser.parse_args()
    
    # Verify scripts exist
    missing = []
    for script in [COLLECT_SCRIPT, FINETUNE_SCRIPT, VALIDATE_SCRIPT]:
        if not script.exists():
            missing.append(script)
    
    if missing:
        print("❌ 錯誤: 找不到必要的腳本:")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)
    
    if args.status:
        show_status()
    else:
        run_pipeline(skip_feedback=args.skip_feedback)


if __name__ == "__main__":
    main()
