import re


def name_convert_to_camel(name: str) -> str:
    """下划线转驼峰(小驼峰)"""
    return re.sub(r'(_[a-z])', lambda x: x.group(1)[1].upper(), name)


def name_convert_to_snake(name: str) -> str:
    """驼峰转下划线"""
    if '_' not in name:
        name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    else:
        raise ValueError(f'{name}字符中包含下划线，无法转换')
    return name.lower()


def name_convert(name: str) -> str:
    """驼峰式命名和下划线式命名互转"""
    is_camel_name = True  # 是否为驼峰式命名
    if '_' in name and re.match(r'[a-zA-Z_]+$', name):
        is_camel_name = False
    elif re.match(r'[a-zA-Z]+$', name) is None:
        raise ValueError(f'Value of "name" is invalid: {name}')
    return name_convert_to_snake(name) if is_camel_name else name_convert_to_camel(name)


def equalsIgnoreCase(a, b):
    if isinstance(a, str):
        if isinstance(b, str):
            return len(a) == len(b) and a.upper() == b.upper()
    return False


if __name__ == '__main__':
    print(name_convert('HelloWorldWorld'))  # -> hello_world_world
    print(name_convert('worker_page_water_hbv'))  # -> woShiOsin
