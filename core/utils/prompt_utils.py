import os

def load_prompt_template(file_path, **kwargs):
    """
    加载提示词模板并替换占位符。

    :param file_path: 模板文件路径
    :param kwargs: 要替换的占位符及其值
    :return: 替换后的提示词字符串
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt template file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        template = file.read()
    
    # 替换占位符
    for key, value in kwargs.items():
        placeholder = f"{{{key}}}"  # 占位符格式为 {key}
        template = template.replace(placeholder, value)
    
    return template