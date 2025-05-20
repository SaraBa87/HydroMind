from setuptools import setup, find_packages

setup(
    name="mutil_tool_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "google-cloud-aiplatform",
        "mysql-connector-python",
        "pandas",
        "numpy",
        "scikit-learn",
    ],
) 