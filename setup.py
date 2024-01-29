# Define Metadata

from setuptools import setup, find_packages

setup(
    name='packageforNIDS',
    version='0.1.0',
    license='MIT',
    packages=find_packages(),
    install_requires=
    ['joblib==1.3.2', 'matplotlib==3.7.0','mlflow==2.2.1','numpy==1.23.5','pandas==1.4.2',
     'pyod==1.1.0','pysad==0.2.0', 'scikit_learn==1.3.2','seaborn==0.11.2', 'tailer==0.4.1', 'combo==0.1.3' ] # List of dependencies
)





