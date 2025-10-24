#!/bin/bash

# Script to create symlinks and wrappers for visual_mapping tools

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_dir> <build_dir>"
    exit 1
fi

SOURCE_DIR=$1
BUILD_DIR=$2

# Create symlinks for the tools, models, and configs
rm -rf ${BUILD_DIR}/bin
rm -rf ${BUILD_DIR}/models
rm -rf ${BUILD_DIR}/configs
ln -sf ${BUILD_DIR}/tools ${BUILD_DIR}/bin
ln -sf ${SOURCE_DIR}/models ${BUILD_DIR}/models
ln -sf ${SOURCE_DIR}/configs ${BUILD_DIR}/configs

echo "Created symlinks for tools, models, and configs at ${BUILD_DIR}/bin, ${BUILD_DIR}/models, and ${BUILD_DIR}/configs"

# Create a wrapper script for cusfm_cli in the build directory
cat > ${BUILD_DIR}/bin/cusfm_cli << 'EOF'
#!/usr/bin/env python3
# This is an auto-generated wrapper script for cusfm_cli

import sys
import os

# Add the package directory to Python path
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, package_dir)

from visual_mapping.cusfm_cli import main

if __name__ == "__main__":
    main()
EOF

# Make the wrapper script executable
chmod +x ${BUILD_DIR}/bin/cusfm_cli

echo "Created executable wrapper at ${BUILD_DIR}/bin/cusfm_cli"
