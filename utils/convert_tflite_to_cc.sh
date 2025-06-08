#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-tflite-model> [out-directory]"
  exit 1
fi

MODEL_PATH="$1"
OUT_DIR="${2:-.}"
MODEL_NAME="my_model"
CC_FILE="${OUT_DIR}/${MODEL_NAME}.cc"
H_FILE="${OUT_DIR}/${MODEL_NAME}.h"

# Ensure model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: File '$MODEL_PATH' not found!"
  exit 1
fi

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# Generate .cc file
echo "#include \"${MODEL_NAME}.h\"" > "$CC_FILE"
echo "" >> "$CC_FILE"

xxd -i "$MODEL_PATH" \
  | sed "s/^unsigned char .* =/const unsigned char ${MODEL_NAME}[] =/" \
  | sed "s/^unsigned int .* =/const unsigned int ${MODEL_NAME}_len =/" >> "$CC_FILE"

# Generate .h file
cat > "$H_FILE" <<EOF
#ifndef C_MODEL_H
#define C_MODEL_H

extern const unsigned char ${MODEL_NAME}[];
extern const unsigned int ${MODEL_NAME}_len;

#endif
EOF

echo "Generated: $CC_FILE and $H_FILE"
