#!/usr/bin/env python3
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

try:
    from qa_evaluator import get_qa_evaluator
    evaluator = get_qa_evaluator()
    evaluator.print_terminal_report()
except ImportError as e:
    print(f"Error: Could not import evaluator. Make sure you are in the backend directory. {e}")
except Exception as e:
    print(f"Error: {e}")
