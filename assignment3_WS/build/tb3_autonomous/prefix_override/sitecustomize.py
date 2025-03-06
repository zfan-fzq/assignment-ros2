import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/zfan/assignment3/assignment3_WS/install/tb3_autonomous'
