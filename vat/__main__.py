"""
VAT命令行入口
"""
from .cli.commands import cli


def main():
    """主函数"""
    cli(obj={})


if __name__ == '__main__':
    main()
