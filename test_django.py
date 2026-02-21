import sys
print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import django
    print("Django version:", django.__version__)
    print("Django import successful!")
except ImportError as e:
    print("Django import failed:", e)
