from setuptools import setup, find_packages

setup(
    name="capcheck",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "python-telegram-bot>=20.8",
        "python-dotenv>=1.0.0",
    ],
    author="CapCheck Team",
    author_email="example@example.com",
    description="A Telegram bot for fact-checking using AI consensus",
    keywords="telegram, bot, fact-checking, consensus, ai",
) 