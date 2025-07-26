from setuptools import setup, find_packages

setup(
    name='quant_trading_system',
    version='0.1.0',
    description='Quantitative trading system with backtesting and Streamlit UI',
    packages=find_packages(where='quant_trading_project'),
    package_dir={'': 'quant_trading_project'},
    install_requires=[
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'requests>=2.28.0',
        'yfinance>=0.2.0',
        'scikit-learn>=1.1.0',
        'lightgbm>=3.3.0',
        'backtesting>=0.3.3',
        'alpaca-py>=0.8.0',
        'streamlit>=1.25.0',
        'urllib3>=1.26.0',
        'finnhub-python>=2.4.0',
        'python-dotenv>=0.19.0',
        'plotly>=5.0.0'
    ],
)
