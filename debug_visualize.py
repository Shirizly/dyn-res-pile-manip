#!/usr/bin/env python
# Debug wrapper for visualize_mpc.py
import sys
import faulthandler
import traceback

# Enable faulthandler to print stack trace on crash
faulthandler.enable()

# Increase verbosity
import warnings
warnings.filterwarnings('default')

print("=" * 60)
print("DEBUG MODE: Running visualize_mpc.py with diagnostics")
print("=" * 60)

try:
    # Import and run main
    from visualize_mpc import main
    print("\n[DEBUG] Starting main()...")
    main()
    print("\n[DEBUG] main() completed successfully")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("PYTHON EXCEPTION CAUGHT:")
    print("=" * 60)
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
