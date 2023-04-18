def convert_byte_int(byte):
    integer = int(byte)
    integer += 256
    if integer >= 256:
        integer -= 256
    return integer
