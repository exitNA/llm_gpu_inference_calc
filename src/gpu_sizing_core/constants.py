PRECISION_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

BYTES_PER_KIB = 1024.0
BYTES_PER_MIB = BYTES_PER_KIB * 1024.0
BYTES_PER_GIB = BYTES_PER_MIB * 1024.0

# In this project, vendor VRAM labels such as 80GB / 96GB / 141GB
# are interpreted as binary-capacity units for sizing.
BYTES_PER_VENDOR_VRAM_GB = BYTES_PER_GIB

SECONDS_PER_DAY = 24 * 60 * 60
