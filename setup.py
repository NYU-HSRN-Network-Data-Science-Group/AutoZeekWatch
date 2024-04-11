from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='AutoZeekWatch',
    version='0.1.1',
    license='MIT',
    description="Network Intrusion Detection using Zeek logs",
    url="https://github.com/zoe70416/NIDS/",
    project_urls={
        'Documentation': 'https://github.com/zoe70416/NIDS',
        'Bug Reports': 'https://github.com/zoe70416/NIDS/issues',
        'Source Code': 'https://github.com/zoe70416/NIDS',
    },
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=
    ['joblib==1.3.2', 'matplotlib==3.7.0','mlflow==2.2.1','numpy==1.23.5','pandas==1.4.2',
     'pyod==1.1.0','pysad==0.2.0', 'scikit_learn==1.3.2','seaborn==0.11.2', 'tailer==0.4.1', 'combo==0.1.3' ] # List of dependencies
)

